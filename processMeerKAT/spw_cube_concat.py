#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

# SPW-cube concatenation step. Runs as a single SLURM job that depends on the whole
# science_image job array (one task per SPW). It merges the per-SPW full-Stokes MFS images
# written into SPW_MFSs/ by that array into one 4D (RA, Dec, Stokes, freq) cube plus a
# companion freqfile.dat, and optionally smooths every slice to a common beam. The actual
# merge/beam/freqfile logic lives in science_image.py so there's a single implementation;
# this script is just the driver that the pipeline schedules after the imaging array.

import os

import config_parser
import bookkeeping

import logging
from time import gmtime
logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s")

# Importing science_image pulls in casatasks + the shared helpers (_resolve_spws,
# _concat_spw_cube) and the module-level ia/imsmooth they use.
import science_image as si


def spw_cube_concat(vis, deconvolver, keepmms, spwid='', common_beam=False, **kwargs):
    visbase = os.path.split(vis.rstrip('/ '))[1]
    extn = '.ms' if keepmms == False else '.mms'
    imagebase = visbase.replace(extn, '.science_image')

    outdir = 'SPW_MFSs'
    spw_ids, labels, central_freqs = si._resolve_spws(vis, spwid)
    imagenames = [os.path.join(outdir, '{0}.{1}.{2}'.format(sid, label, imagebase))
                  for sid, label in zip(spw_ids, labels)]

    cubename = os.path.join(outdir, imagebase + '.cube.image')
    logger.info("Assembling SPW cube from {0} per-SPW image(s) -> {1}".format(len(imagenames), cubename))
    si._concat_spw_cube(imagenames, central_freqs, cubename, deconvolver, common_beam=common_beam)


if __name__ == '__main__':

    args, params = bookkeeping.get_imaging_params()
    spw_cube_concat(**params)
    bookkeeping.rename_logs(si.logfile)
