#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

import os
import numpy as np

import config_parser
import bookkeeping

from casatasks import *
logfile=casalog.logfile()
casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))
import casampi

import logging
from time import gmtime
logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s")

import shutil
from katbeam import JimBeam
from casatools import image, msmetadata
ia = image()


def _versioned(path):
    """Return path unchanged if it doesn't exist, else path_2, path_3, ..."""
    if not os.path.exists(path):
        return path
    version = 2
    while True:
        candidate = '{0}_{1}'.format(path, version)
        if not os.path.exists(candidate):
            return candidate
        version += 1


def _unmask_all(imagepath):
    """Strip any inherited mask so downstream immath sees every pixel as valid."""
    ia.open(imagepath)
    ia.calcmask("T")
    ia.close()


def _get_restoring_beam(imagepath):
    """Return a single restoring-beam dict {major,minor,positionangle} from imagepath, or None.

    The PSF image (.psf.tt0) carries the fitted beam even when residual-derived
    products don't, so it's the most reliable source. Handles both the single-beam
    form and the per-plane ('beams') form returned for multi-Stokes images."""
    if not os.path.exists(imagepath):
        return None
    ia.open(imagepath)
    try:
        beam = ia.restoringbeam()
    finally:
        ia.close()
    if not beam:
        return None
    if 'beams' in beam:
        # Per-plane beams (e.g. a multi-Stokes cube): take the first available plane.
        chan = sorted(beam['beams'].keys())[0]
        pol = sorted(beam['beams'][chan].keys())[0]
        beam = beam['beams'][chan][pol]
    if 'major' not in beam:
        return None
    return beam


def _set_restoring_beam(target, beam):
    """Stamp a single restoring beam onto an existing CASA image so PyBDSF/FITS can read it."""
    if beam is None or not os.path.exists(target):
        return
    major = '{0}{1}'.format(beam['major']['value'], beam['major']['unit'])
    minor = '{0}{1}'.format(beam['minor']['value'], beam['minor']['unit'])
    pa    = '{0}{1}'.format(beam['positionangle']['value'], beam['positionangle']['unit'])
    ia.open(target)
    try:
        ia.setrestoringbeam(major=major, minor=minor, pa=pa)
    finally:
        ia.close()


def _stamp_beam_from_psf(imagename, deconvolver):
    """Copy the fitted restoring beam from the PSF onto the restored brightness products.

    In parallel / multi-Stokes mtmfs runs CASA fits the beam into .psf.tt0 but often leaves
    the restored .image.tt* (and .residual.tt*) WITHOUT a global restoring beam, so CARTA,
    PyBDSF and FITS export have no beam to read/draw. We take the beam from .psf.tt0 (the
    trusted fitted beam) and stamp the same single global beam onto every Jy/beam product so
    they're all identical. Products that aren't brightness maps (.model, .pb, .sumwt) are left
    untouched — a restoring beam is meaningless there (model is Jy/pixel, pb/sumwt are
    dimensionless), and CARTA doesn't need one to display them."""
    psf = imagename + ('.psf.tt0' if deconvolver == 'mtmfs' else '.psf')
    beam = _get_restoring_beam(psf)
    if beam is None:
        logger.warning("No restoring beam in '{0}'; cannot stamp restored products.".format(psf))
        return
    if deconvolver == 'mtmfs':
        targets = [imagename + '.image.tt0', imagename + '.image.tt1',
                   imagename + '.residual.tt0', imagename + '.residual.tt1']
    else:
        targets = [imagename + '.image', imagename + '.residual']
    for t in targets:
        if os.path.exists(t) and _get_restoring_beam(t) is None:
            _set_restoring_beam(t, beam)
            logger.info("Stamped restoring beam from {0} onto {1}".format(
                os.path.basename(psf), os.path.basename(t)))


def make_alpha(imagename, deconvolver, stokes, alpha_nsigma=1.0):
    """
    Build a spectral-index (alpha) image and its error map from an mtmfs run.

    Skipped if deconvolver != 'mtmfs' (no Taylor terms) or if stokes is 'I' alone
    (CASA's tclean already auto-writes .alpha + .alpha.error in that case).

    For multi-Stokes runs (e.g. 'IQUV') we extract the Stokes I plane from each
    Taylor-term image first, since alpha is fundamentally a Stokes-I quantity.

    Produces just the two raw maps:
      1. alpha       = tt1 / tt0
      2. alpha_error = sqrt((resid_tt1/tt0)^2 + (alpha * resid_tt0/tt0)^2)

    Any noise thresholding / sigma clipping is intentionally left to the user to do
    separately on these maps. (alpha_nsigma is accepted for backward compatibility but
    is no longer used.)
    """
    if deconvolver != 'mtmfs':
        logger.info("Skipping alpha image: deconvolver='{0}' (mtmfs required).".format(deconvolver))
        return
    if stokes.upper() == 'I':
        # CASA's tclean already wrote .alpha + .alpha.error during restoration.
        return

    img0   = imagename + '.image.tt0'
    img1   = imagename + '.image.tt1'
    resid0 = imagename + '.residual.tt0'
    resid1 = imagename + '.residual.tt1'
    psf0   = imagename + '.psf.tt0'

    for f in [img0, img1, resid0, resid1]:
        if not os.path.exists(f):
            logger.warning("Skipping alpha image: '{0}' not found.".format(f))
            return

    # Restoring beam for the alpha products: the residual-derived error map carries no beam,
    # so read it from .psf.tt0 (falling back to the restored Stokes-I image) and stamp it
    # explicitly onto both maps so CARTA/FITS can always read BMAJ/BMIN/BPA.
    beam = _get_restoring_beam(psf0) or _get_restoring_beam(img0)
    if beam is None:
        logger.warning("No restoring beam found in '{0}' or '{1}'; alpha products may lack beam info.".format(psf0, img0))

    # Extract the Stokes I plane from each Taylor-term image — alpha is a Stokes-I concept.
    work = {}
    for label, src in [('img0', img0), ('img1', img1), ('resid0', resid0), ('resid1', resid1)]:
        si = src + '.StokesI'
        if not os.path.exists(si):
            _extract_stokesI(src, si)  # ia toolkit, not imsubimage — no MPI soft-hang
        work[label] = si

    alpha_out     = _versioned(imagename + '.alpha.image')
    alpha_err_out = _versioned(imagename + '.alpha.error.image')

    logger.info("Building alpha map -> {0}".format(os.path.basename(alpha_out)))
    immath(imagename=[work['img1'], work['img0']], expr='IM0/IM1', outfile=alpha_out)
    _unmask_all(alpha_out)
    _set_restoring_beam(alpha_out, beam)

    logger.info("Building alpha error map -> {0}".format(os.path.basename(alpha_err_out)))
    immath(imagename=[work['resid1'], work['img1'], work['img0'], work['resid0']],
           expr='sqrt((IM0/IM2)^2 + (((IM1/IM2)*IM3/IM2)^2))',
           outfile=alpha_err_out,
           imagemd=work['img0'])  # Inherit coordsys from restored Stokes I (residuals carry no restoring beam)
    _unmask_all(alpha_err_out)
    _set_restoring_beam(alpha_err_out, beam)


def do_pb_corr(inpimage, pbthreshold=0, pbband='LBand'):
    """
    Given the input CASA image, outputs a katbeam corrected image, optionally
    cutoff at a specified threshold.

    Inputs:
    inpimage        Input CASA image name, str
    pbthreshold     Cutoff threshold to mask the PB, float
    pbband          Band at which to generate the PB

    Outputs:
    None
    """

    pbcorimage = inpimage.replace('.image', '.katbeam_pbcor.image')
    pbimage = inpimage.replace('.image', '.katbeam.pb')

    ia.open(inpimage)
    csys = ia.coordsys().torecord()
    imgdata = ia.getchunk()
    shape = ia.shape()
    ia.close()

    cx, cy = shape[0]//2, shape[1]//2

    # Size of each pixel
    cdelt = np.abs(csys['direction0']['cdelt'][0])
    unit = csys['direction0']['units'][0]

    if unit == 'rad':
        cdelt = np.rad2deg(cdelt)
    elif unit == "'": #arcmin
        cdelt /= 60.

    # Frequency of image, convert from Hz to MHz
    try:
        freq = csys['spectral1']['wcs']['crval']/1e6
    except KeyError:
        freq = csys['spectral2']['wcs']['crval']/1e6

    if pbband == 'LBand':
        PBeam = JimBeam('MKAT-AA-L-JIM-2020')
    elif pbband == 'SBand':
        PBeam = JimBeam('MKAT-AA-S-JIM-2020')
    elif pbband == 'UHF':
        PBeam = JimBeam('MKAT-AA-UHF-JIM-2020')
    else:
        logger.error('Input pbband not recognized. Must be one of LBand, SBand or UHF. Defaulting to LBand.')
        PBeam = JimBeam('MKAT-AA-L-JIM-2020')

    x = np.linspace(-cx, cx+1, shape[0])
    y = np.linspace(-cy, cy+1, shape[1])

    xx, yy = np.meshgrid(x, y)

    # Convert pixels into separation in degrees
    xx *= cdelt
    yy *= cdelt

    # Generate the 2D PB image
    beam_I = PBeam.I(xx, yy, freq)

    # Match shape with image data for PB correction
    if len(shape) == 4:
        beam_I = beam_I[:, :, None, None]

    pbcor_imgdata = imgdata/beam_I

    # Mask below the threshold
    if pbthreshold > 0:
        pbcor_imgdata[beam_I < pbthreshold] = np.nan
        #beam_I[beam_I < pbthreshold] = np.nan

    shutil.copytree(inpimage, pbimage)
    ia.open(pbimage)
    ia.putchunk(beam_I)
    ia.close()

    shutil.copytree(inpimage, pbcorimage)
    ia.open(pbcorimage)
    ia.putchunk(pbcor_imgdata)
    ia.close()


def _extract_stokesI(imname, outname):
    """Extract the Stokes-I plane using only the ia/rg toolkit (in-process, serial).

    This deliberately avoids the imsubimage CASA task: under mpirun/casampi that task
    reaches into the MPI framework and, on a parallel-tclean multi-Stokes image, can
    soft-hang (rank 0 waits indefinitely). The toolkit calls below all run in-process on
    the rank executing the script, with no MPI dispatch, so there is nothing to stall on."""
    from casatools import regionmanager
    rg = regionmanager()
    ia.open(imname)
    csys = ia.coordsys()
    shape = list(ia.shape())
    # Locate the Stokes axis and the pixel index of Stokes I robustly (don't assume IQUV order).
    try:
        sax = csys.findaxisbyname('Stokes')
    except Exception:
        sax = 2
    try:
        i_idx = list(csys.stokes()).index('I')
    except Exception:
        i_idx = 0
    blc = [0] * len(shape)
    trc = [s - 1 for s in shape]
    blc[sax] = i_idx
    trc[sax] = i_idx
    reg = rg.box(blc=blc, trc=trc)
    sub = ia.subimage(outfile=outname, region=reg, dropdeg=False, overwrite=True)
    sub.done()
    csys.done()
    ia.close()
    rg.done()


def _parse_spwid(spwid):
    """Normalise spwid to a list of int SPW IDs, accepting either form:
    a list ([0,1,2]) or a comma-separated string ('0,1,2'). Empty ('' / [] / None) -> []."""
    if spwid is None:
        return []
    if isinstance(spwid, (list, tuple)):
        return [int(s) for s in spwid]
    s = str(spwid).strip().strip('[]')
    return [int(x) for x in s.split(',') if x.strip() != '']


def _resolve_spws(vis, spwid):
    """Return (spw_ids, freq_labels, central_freqs) for the SPWs to image.

    spw_ids are the integer SPW IDs in the MS; freq_labels are 'LO-HIMHz' strings
    derived from each SPW's channel frequency range (so there's no parallel
    freq-band list to maintain); central_freqs are the mean channel frequency (Hz)
    of each SPW, used to label the cube slices. spwid may be a list ([0,1,2]) or a
    comma-separated string ('0,1,2'); if empty, every SPW is used.
    """
    msmd = msmetadata()
    msmd.open(vis)
    try:
        requested = _parse_spwid(spwid)
        if requested:
            ids = requested
        else:
            ids = list(range(msmd.nspw()))
        labels = []
        central_freqs = []
        for sid in ids:
            freqs = msmd.chanfreqs(sid)  # Hz
            labels.append('{0:.0f}-{1:.0f}MHz'.format(freqs.min() / 1e6, freqs.max() / 1e6))
            central_freqs.append(float(freqs.mean()))  # SPW central (mean channel) frequency, Hz
    finally:
        msmd.done()
    return ids, labels, central_freqs


def _concat_spw_cube(imagenames, central_freqs, outname, deconvolver, common_beam=False):
    """Merge the per-SPW full-Stokes MFS images into one 4D (RA, Dec, Stokes, freq) cube.

    Each per-SPW image is a single-frequency, full-Stokes plane at its own SPW frequency and
    carries its own restoring beam (stamped from that SPW's .psf.tt0). ia.imageconcat stitches
    them along the spectral axis into a single paged cube on disk. Because the input beams
    differ between SPWs, CASA writes a PER-PLANE BEAM TABLE into the cube automatically — so
    CARTA shows the correct (frequency-dependent) beam on every channel. reorder=True sorts the
    planes by frequency; relax=True tolerates the non-uniform SPW spacing left after flagging.

    central_freqs (Hz, parallel to imagenames) are the true SPW mean frequencies. Because the
    SPW spacing is non-uniform, the exact per-slice frequency is written to a companion
    <base>.cube.freqfile.dat (one value per slice, ascending, matching the cube's channel order)
    so the user always knows which frequency each slice corresponds to, independent of the
    cube's (possibly linearised) spectral WCS.

    If common_beam is True, the cube is additionally convolved to a single common beam (the
    smallest beam enclosing every per-plane beam) and written to <base>.cube.commonbeam.image,
    so all slices share one resolution instead of the per-plane beam table.

    Returns the path of the primary cube written, or None if there aren't >=2 SPW images."""
    suffix = '.image.tt0' if deconvolver == 'mtmfs' else '.image'
    # Pair each existing image with its central frequency, then sort ascending so the file
    # order matches imageconcat's reorder=True (which sorts slices by frequency).
    pairs = sorted((f, n + suffix) for n, f in zip(imagenames, central_freqs)
                   if os.path.exists(n + suffix))
    if len(pairs) < 2:
        logger.warning("spw_cube: found {0} SPW image(s); need >=2 to build a cube. Skipping concat.".format(len(pairs)))
        return None
    sorted_freqs = [f for f, _ in pairs]
    infiles = [n for _, n in pairs]

    # Companion frequency list: one central frequency (Hz) per slice, ascending.
    freqfile = outname.replace('.cube.image', '.cube.freqfile.dat')
    with open(freqfile, 'w') as fh:
        for f in sorted_freqs:
            fh.write('{0:.10e}\n'.format(f))
    logger.info("Wrote per-slice central frequencies -> {0}".format(freqfile))

    if os.path.exists(outname):
        shutil.rmtree(outname)
    logger.info("Concatenating {0} per-SPW images into frequency cube -> {1}".format(len(infiles), outname))
    cube = ia.imageconcat(outfile=outname, infiles=infiles, axis=-1, relax=True,
                          reorder=True, overwrite=True, mode='paged')

    if common_beam:
        # Smooth every channel to one common resolution: commonbeam() returns the smallest
        # beam that encloses all per-plane beams, and imsmooth(targetres=True) convolves each
        # plane up to exactly that beam. The per-plane-beam cube ('outname') is kept as well.
        cbeam = cube.commonbeam()
        cube.done()
        major = '{0}{1}'.format(cbeam['major']['value'], cbeam['major']['unit'])
        minor = '{0}{1}'.format(cbeam['minor']['value'], cbeam['minor']['unit'])
        pa    = '{0}{1}'.format(cbeam['pa']['value'], cbeam['pa']['unit'])
        smoothed = outname.replace('.cube.image', '.cube.commonbeam.image')
        if os.path.exists(smoothed):
            shutil.rmtree(smoothed)
        logger.info("common_beam=True: smoothing cube to common beam {0} x {1} @ {2} -> {3}".format(
            major, minor, pa, smoothed))
        imsmooth(imagename=outname, kernel='gauss', major=major, minor=minor, pa=pa,
                 targetres=True, outfile=smoothed)
        return smoothed

    cube.done()
    return outname


def _products_complete(imagename, deconvolver, stokes, pbcorr):
    """True if every expected final product for this image already exists, so the whole step
    can be skipped and the job can end as a clean, complete run (no re-imaging, no duplicate
    _2/_3 alpha maps). Mirrors exactly what _build_and_clean would produce for these settings."""
    imname = imagename + ('.image.tt0' if deconvolver == 'mtmfs' else '.image')
    if not os.path.exists(imname):
        return False
    # Alpha maps: produced only for multi-/non-I mtmfs runs (CASA auto-writes them for plain I).
    if deconvolver == 'mtmfs' and stokes.upper() != 'I':
        if not (os.path.exists(imagename + '.alpha.image') and os.path.exists(imagename + '.alpha.error.image')):
            return False
    # katbeam PB correction: only when explicitly enabled (pbcorr=True) and Stokes I present.
    if pbcorr and 'I' in stokes.upper():
        multi = len(stokes) > 1 and 'I' in stokes.upper()
        base = (imname + '.StokesI') if multi else imname
        if not os.path.exists(base.replace('.image', '.katbeam_pbcor.image')):
            return False
    return True


def _build_and_clean(vis, imagename, spw, cell, robust, imsize, wprojplanes, niter, threshold, multiscale, nterms,
                     gridder, deconvolver, restoringbeam, stokes, mask, outlierfile, pbthreshold, pbband,
                     usemask, sidelobethreshold, noisethreshold, lownoisethreshold, negativethreshold, alpha_nsigma,
                     pbcorr=False):
    """Run tclean (+ alpha + PB correction) for a single image / SPW selection.

    spw is a tclean spw-selection string ('' for the whole band, or e.g. '0')."""

    # If everything this step would produce is already on disk, do nothing — this makes a rerun
    # a fast, clean "complete" pass instead of redoing (or re-hanging on) finished work.
    if _products_complete(imagename, deconvolver, stokes, pbcorr):
        logger.info('All expected products for "{0}" already exist — nothing to do, skipping.'.format(imagename))
        return

    if deconvolver == 'mtmfs':
        imname = imagename + '.image.tt0'
    else:
        imname = imagename + '.image'

    if not os.path.exists(imname):

        tclean_kwargs = dict(
            vis=vis, selectdata=False, datacolumn='corrected', imagename=imagename, spw=spw,
            imsize=imsize, cell=cell, stokes=stokes, gridder=gridder, specmode='mfs',
            wprojplanes=wprojplanes, deconvolver=deconvolver, restoration=True,
            weighting='briggs', robust=robust, niter=niter, scales=multiscale,
            threshold=threshold, nterms=nterms, calcpsf=True, outlierfile=outlierfile,
            pblimit=-1, restoringbeam=restoringbeam, parallel=True,
        )

        if usemask == 'auto-multithresh':
            logger.info("Using auto-multithresh masking with sidelobethreshold={0}, noisethreshold={1}, lownoisethreshold={2}, negativethreshold={3}".format(
                sidelobethreshold, noisethreshold, lownoisethreshold, negativethreshold))
            tclean_kwargs.update(dict(
                usemask='auto-multithresh',
                mask='',
                sidelobethreshold=sidelobethreshold,
                noisethreshold=noisethreshold,
                lownoisethreshold=lownoisethreshold,
                negativethreshold=negativethreshold,
            ))
        else:
            tclean_kwargs.update(dict(usemask='user', mask=mask))

        tclean(**tclean_kwargs)

    else:
        logger.warning('Output image "{0}" already exists. Skipping tclean step and applying pb correction.'.format(imname))

    # Parallel/multi-Stokes mtmfs often leaves the restored .image.tt* without a global beam,
    # even though .psf.tt0 carries it — so CARTA/PyBDSF/FITS see no beam. Stamp the psf beam
    # onto the restored brightness products here, BEFORE the StokesI subimage + PB correction
    # below, so those derivatives inherit the same beam.
    _stamp_beam_from_psf(imagename, deconvolver)

    # Produce a spectral-index image only when polarisation imaging is involved
    # (stokes != 'I'); CASA already auto-writes alpha+error for plain Stokes-I mtmfs runs.
    make_alpha(imagename, deconvolver, stokes, alpha_nsigma=alpha_nsigma)

    # katbeam primary-beam correction (produces .katbeam.pb + .katbeam_pbcor.image) is OFF by
    # default in this fork — it is slow on large multi-Stokes images and most users apply their
    # own PB correction. Enable it with pbcorr=True in the [image] config section.
    if not pbcorr:
        logger.info('pbcorr=False: skipping katbeam primary-beam correction (no .katbeam* products written).')
    else:
        if len(stokes) > 1 and 'I' in stokes.upper():
            logger.warning('Output image "{0}" includes multiple Stokes, but katbeam only applicable to Stokes I. Selecting Stokes I (via ia toolkit, no MPI task) and applying PB correction.'.format(imname))
            stokesI = imname + '.StokesI'
            if not os.path.exists(stokesI):
                _extract_stokesI(imname, stokesI)
            imname = stokesI

        if 'I' in stokes.upper():
            do_pb_corr(imname, pbthreshold, pbband)


def science_image(vis, cell, robust, imsize, wprojplanes, niter, threshold, multiscale, nterms, gridder, deconvolver, restoringbeam, stokes, mask, rmsmap, outlierfile, keepmms, pbthreshold, pbband,
                  usemask='user', sidelobethreshold=0.5, noisethreshold=5.0, lownoisethreshold=0.01, negativethreshold=0.0,
                  alpha_nsigma=1.0, spw_cube=False, common_beam=False, spwid='', pbcorr=False):

    visbase = os.path.split(vis.rstrip('/ '))[1] # Get only vis name, not entire path
    extn = '.ms' if keepmms==False else '.mms'
    imagebase = visbase.replace(extn, '.science_image') # Images will be produced in $CWD

    if os.path.exists(outlierfile) and open(outlierfile).read() == '':
        outlierfile = ''

    if not (type(threshold) is str and 'Jy' in threshold) and threshold > 1 and os.path.exists(rmsmap):
        stats = imstat(imagename=rmsmap)
        threshold *= stats['min'][0]

    common = dict(cell=cell, robust=robust, imsize=imsize, wprojplanes=wprojplanes, niter=niter,
                  threshold=threshold, multiscale=multiscale, nterms=nterms, gridder=gridder,
                  deconvolver=deconvolver, restoringbeam=restoringbeam, stokes=stokes, mask=mask,
                  outlierfile=outlierfile, pbthreshold=pbthreshold, pbband=pbband, usemask=usemask,
                  sidelobethreshold=sidelobethreshold, noisethreshold=noisethreshold,
                  lownoisethreshold=lownoisethreshold, negativethreshold=negativethreshold,
                  alpha_nsigma=alpha_nsigma, pbcorr=pbcorr)

    if spw_cube:
        spw_ids, labels, central_freqs = _resolve_spws(vis, spwid)
        outdir = 'SPW_MFSs'
        os.makedirs(outdir, exist_ok=True)

        # When launched as one task of the science_image SLURM job array (built by
        # processMeerKAT.py when spw_cube=True), each task images exactly ONE SPW so all SPWs
        # image concurrently on separate nodes. The cube is then assembled by the separate
        # spw_cube_concat.py job that depends on the whole array. SLURM_ARRAY_TASK_ID selects
        # the SPW. If not running in an array (e.g. a manual/serial run), fall back to imaging
        # every SPW in a loop and concatenating here.
        task = os.environ.get('SLURM_ARRAY_TASK_ID')
        if task is not None:
            i = int(task)
            if i >= len(spw_ids):
                logger.warning("Array task {0} >= number of SPWs ({1}); nothing to image.".format(i, len(spw_ids)))
                return
            sid, label = spw_ids[i], labels[i]
            imagename = os.path.join(outdir, '{0}.{1}.{2}'.format(sid, label, imagebase))
            logger.info("Array task {0}: imaging SPW {1} ({2}) -> {3}".format(i, sid, label, imagename))
            _build_and_clean(vis, imagename, str(sid), **common)
            return  # cube assembly is handled by the dependent spw_cube_concat.py job

        logger.info("spw_cube=True (serial): imaging {0} SPW(s) {1} into '{2}/'.".format(len(spw_ids), spw_ids, outdir))
        imagenames = []
        for sid, label in zip(spw_ids, labels):
            imagename = os.path.join(outdir, '{0}.{1}.{2}'.format(sid, label, imagebase))
            logger.info("Imaging SPW {0} ({1}) -> {2}".format(sid, label, imagename))
            _build_and_clean(vis, imagename, str(sid), **common)
            imagenames.append(imagename)
        cubename = os.path.join(outdir, imagebase + '.cube.image')
        _concat_spw_cube(imagenames, central_freqs, cubename, deconvolver, common_beam=common_beam)
    else:
        _build_and_clean(vis, imagebase, '', **common)

if __name__ == '__main__':

    args,params = bookkeeping.get_imaging_params()
    science_image(**params)
    bookkeeping.rename_logs(logfile)

    # Ensure the job ALWAYS terminates promptly once the work is done. When tclean is skipped
    # (the image already exists) the casampi MPI servers are started at import but never engaged,
    # and the interpreter can then hang at exit on an MPI teardown handshake that never completes
    # — the job sits idle until the SLURM time limit (observed: 8 s of real work, then a ~2.5 h
    # stall ending in a TIME LIMIT kill). We stop the MPI servers cleanly for a tidy COMPLETED
    # status, but guard the whole thing with a hard timeout: if the teardown itself is what
    # hangs, an alarm fires and forces the exit, so the job can never idle to the time limit.
    import sys, signal

    def _hard_exit(signum=None, frame=None):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    try:
        signal.signal(signal.SIGALRM, _hard_exit)
        signal.alarm(120)  # backstop: never let MPI teardown stall the job beyond 2 minutes
        from casampi.MPIEnvironment import MPIEnvironment
        if getattr(MPIEnvironment, 'is_mpi_enabled', False) and getattr(MPIEnvironment, 'is_mpi_client', False):
            from casampi.MPICommandClient import MPICommandClient
            MPICommandClient().stop_services()
        signal.alarm(0)
    except Exception as e:
        logger.warning('MPI server shutdown skipped/failed ({0}); forcing exit anyway.'.format(e))

    _hard_exit()
