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

try:
    import bdsf
except ImportError:
    bdsf = None


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


def make_alpha(imagename, deconvolver, stokes, alpha_nsigma=1.0):
    """
    Build a noise-thresholded spectral-index (alpha) image from an mtmfs run.

    Skipped if deconvolver != 'mtmfs' (no Taylor terms) or if stokes is 'I' alone
    (CASA's tclean already auto-writes .alpha + .alpha.error in that case).

    For multi-Stokes runs (e.g. 'IQUV') we extract the Stokes I plane from each
    Taylor-term image first, since alpha is fundamentally a Stokes-I quantity.

    Pipeline:
      1. alpha = tt1 / tt0
      2. alpha_error = sqrt((resid_tt1/tt0)^2 + (alpha * resid_tt0/tt0)^2)
      3. PyBDSF on alpha_error to produce a 2D RMS map (rms_map=True)
      4. Clip alpha_error pixel-wise where alpha_error >= 5 * RMS_map
      5. Final alpha mask: keep alpha where |alpha| > alpha_nsigma * clipped alpha_error
    """
    if deconvolver != 'mtmfs':
        logger.info("Skipping alpha image: deconvolver='{0}' (mtmfs required).".format(deconvolver))
        return
    if stokes.upper() == 'I':
        # CASA's tclean already wrote .alpha + .alpha.error during restoration.
        return
    if bdsf is None:
        logger.warning("Skipping alpha image: pybdsf not available in this environment.")
        return

    img0   = imagename + '.image.tt0'
    img1   = imagename + '.image.tt1'
    resid0 = imagename + '.residual.tt0'
    resid1 = imagename + '.residual.tt1'

    for f in [img0, img1, resid0, resid1]:
        if not os.path.exists(f):
            logger.warning("Skipping alpha image: '{0}' not found.".format(f))
            return

    # Extract the Stokes I plane from each Taylor-term image — alpha is a Stokes-I concept.
    work = {}
    for label, src in [('img0', img0), ('img1', img1), ('resid0', resid0), ('resid1', resid1)]:
        si = src + '.StokesI'
        if not os.path.exists(si):
            imsubimage(imagename=src, outfile=si, stokes='I')
        work[label] = si

    alpha_out      = _versioned(imagename + '.alpha.image')
    alpha_err_out  = _versioned(imagename + '.alpha.error.image')
    alpha_err_rms  = imagename + '.alpha.error.rms'
    alpha_clip_out = _versioned(imagename + '.alpha.error.5sigma.clip.image')
    alpha_sig_out  = _versioned(imagename + '.alpha.{0}sigma.image'.format(int(alpha_nsigma)))

    logger.info("Building alpha map -> {0}".format(os.path.basename(alpha_out)))
    immath(imagename=[work['img1'], work['img0']], expr='IM0/IM1', outfile=alpha_out)
    _unmask_all(alpha_out)

    logger.info("Building alpha error map -> {0}".format(os.path.basename(alpha_err_out)))
    immath(imagename=[work['resid1'], work['img1'], work['img0'], work['resid0']],
           expr='sqrt((IM0/IM2)^2 + (((IM1/IM2)*IM3/IM2)^2))',
           outfile=alpha_err_out,
           imagemd=work['img0'])  # Inherit beam/coordsys from restored Stokes I (residuals carry no restoring beam) so PyBDSF can read BMAJ/BMIN/BPA
    _unmask_all(alpha_err_out)

    # PyBDSF wants a FITS file for header stability (matches master's selfcal pattern).
    fitsname = alpha_err_out + '.fits'
    if not os.path.exists(fitsname):
        exportfits(imagename=alpha_err_out, fitsimage=fitsname)

    logger.info("Running PyBDSF for 2D RMS map of alpha error -> {0}".format(os.path.basename(alpha_err_rms)))
    img = bdsf.process_image(fitsname, adaptive_rms_box=True,
        rms_box_bright=(40, 5), advanced_opts=True, mean_map='map',
        rms_box=(100, 30), rms_map=True, thresh='hard', thresh_isl=3.0, thresh_pix=5.0,
        blank_limit=1e-10)
    img.export_image(outfile=alpha_err_rms, img_type='rms', img_format='casa', clobber=True)

    logger.info("Clipping alpha error pixel-wise at 5 x RMS -> {0}".format(os.path.basename(alpha_clip_out)))
    immath(imagename=[alpha_err_out, alpha_err_rms],
           expr='iif(IM0 < 5*IM1, IM0, 0/0)',
           outfile=alpha_clip_out,
           imagemd=work['img0'])

    logger.info("Masking alpha where |alpha| > {0} x clipped error -> {1}".format(alpha_nsigma, os.path.basename(alpha_sig_out)))
    immath(imagename=[alpha_out, alpha_clip_out],
           expr='iif(abs(IM0) > {0}*IM1, IM0, 0/0)'.format(alpha_nsigma),
           outfile=alpha_sig_out,
           imagemd=work['img0'])


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


def _resolve_spws(vis, spwid):
    """Return (spw_ids, freq_labels) for the SPWs to image.

    spw_ids are the integer SPW IDs in the MS; freq_labels are 'LO-HIMHz' strings
    derived from each SPW's channel frequency range (so there's no parallel
    freq-band list to maintain). If spwid is '', every SPW in the MS is used.
    """
    msmd = msmetadata()
    msmd.open(vis)
    try:
        if spwid.strip() != '':
            ids = [int(s) for s in spwid.split(',') if s.strip() != '']
        else:
            ids = list(range(msmd.nspw()))
        labels = []
        for sid in ids:
            freqs = msmd.chanfreqs(sid)  # Hz
            labels.append('{0:.0f}-{1:.0f}MHz'.format(freqs.min() / 1e6, freqs.max() / 1e6))
    finally:
        msmd.done()
    return ids, labels


def _build_and_clean(vis, imagename, spw, cell, robust, imsize, wprojplanes, niter, threshold, multiscale, nterms,
                     gridder, deconvolver, restoringbeam, stokes, mask, outlierfile, pbthreshold, pbband,
                     usemask, sidelobethreshold, noisethreshold, lownoisethreshold, negativethreshold, alpha_nsigma):
    """Run tclean (+ alpha + PB correction) for a single image / SPW selection.

    spw is a tclean spw-selection string ('' for the whole band, or e.g. '0')."""

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

    # Produce a spectral-index image only when polarisation imaging is involved
    # (stokes != 'I'); CASA already auto-writes alpha+error for plain Stokes-I mtmfs runs.
    make_alpha(imagename, deconvolver, stokes, alpha_nsigma=alpha_nsigma)

    if len(stokes) > 1 and 'I' in stokes.upper():
        logger.warning('Output image "{0}" includes multiple Stokes, but katbeam only applicable to Stokes I. Selecting Stokes I and applying PB correction.'.format(imname))
        stokesI = imname + '.StokesI'
        if not os.path.exists(stokesI):
            imsubimage(imagename=imname, outfile=stokesI, stokes='I')
        imname = stokesI

    if 'I' in stokes.upper():
        do_pb_corr(imname, pbthreshold, pbband)


def science_image(vis, cell, robust, imsize, wprojplanes, niter, threshold, multiscale, nterms, gridder, deconvolver, restoringbeam, stokes, mask, rmsmap, outlierfile, keepmms, pbthreshold, pbband,
                  usemask='user', sidelobethreshold=0.5, noisethreshold=5.0, lownoisethreshold=0.01, negativethreshold=0.0,
                  alpha_nsigma=1.0, spw_cube=False, spwid=''):

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
                  alpha_nsigma=alpha_nsigma)

    if spw_cube:
        # Image each spectral window separately into SPWs_full_stokes/ (an SPW "cube" of
        # per-SPW images) instead of producing a single full-bandwidth averaged image.
        spw_ids, labels = _resolve_spws(vis, spwid)
        outdir = 'SPWs_full_stokes'
        os.makedirs(outdir, exist_ok=True)
        logger.info("spw_cube=True: imaging {0} SPW(s) {1} separately into '{2}/'.".format(len(spw_ids), spw_ids, outdir))
        for sid, label in zip(spw_ids, labels):
            imagename = os.path.join(outdir, '{0}.{1}.{2}'.format(sid, label, imagebase))
            logger.info("Imaging SPW {0} ({1}) -> {2}".format(sid, label, imagename))
            _build_and_clean(vis, imagename, str(sid), **common)
    else:
        _build_and_clean(vis, imagebase, '', **common)

if __name__ == '__main__':

    args,params = bookkeeping.get_imaging_params()
    science_image(**params)
    bookkeeping.rename_logs(logfile)
