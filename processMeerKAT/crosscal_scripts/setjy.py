#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

import os, sys, shutil

import bookkeeping
from config_parser import validate_args as va
import numpy as np
import logging
from time import gmtime
logging.Formatter.converter = gmtime

from casatasks import *
logfile=casalog.logfile()
casalog.setlogfile('logs/{SLURM_JOB_NAME}-{SLURM_JOB_ID}.casa'.format(**os.environ))
from casatools import msmetadata
import casampi
msmd = msmetadata()

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)

L_BAND_CALIBRATORS = {
    'J0010-4153': {'flux': 4.545, 'frac_pol': 0.0012, 'pol_angle': -60.2, 'rm': -1.20},
    'J0022+0014': {'flux': 2.900, 'frac_pol': 0.0009, 'pol_angle': -11.3, 'rm': 2.00},
    'J0024-4202': {'flux': 2.871, 'frac_pol': 0.0010, 'pol_angle': -4.1, 'rm': 4.60},
    'J0025-2602': {'flux': 8.731, 'frac_pol': 0.0017, 'pol_angle': 72.8, 'rm': -3.60},
    'J0059+0006': {'flux': 2.449, 'frac_pol': 0.0376, 'pol_angle': 74.3, 'rm': -3.60},
    'J0108+0134': {'flux': 3.113, 'frac_pol': 0.0388, 'pol_angle': -79.7, 'rm': -6.60},
    'J0137+3309': {'flux': 16.112, 'frac_pol': 0.0064, 'pol_angle': -36.4, 'rm': -55.40},
    'J0155-4048': {'flux': 2.161, 'frac_pol': 0.0007, 'pol_angle': 29.9, 'rm': 0.00},
    'J0203-4349': {'flux': 2.726, 'frac_pol': 0.0091, 'pol_angle': -55.5, 'rm': -8.20},
    'J0210-5101': {'flux': 3.402, 'frac_pol': 0.0124, 'pol_angle': 30.7, 'rm': 14.60},
    'J0238+1636': {'flux': 0.528, 'frac_pol': 0.0077, 'pol_angle': 19.6, 'rm': 48.00},
    'J0240-2309': {'flux': 5.938, 'frac_pol': 0.0098, 'pol_angle': -42.2, 'rm': 10.00},
    'J0252-7104': {'flux': 5.811, 'frac_pol': 0.0018, 'pol_angle': -35.3, 'rm': 2.40},
    'J0303-6211': {'flux': 3.194, 'frac_pol': 0.0471, 'pol_angle': 46.6, 'rm': 49.60},
    'J0318+1628': {'flux': 7.619, 'frac_pol': 0.0003, 'pol_angle': -19.9, 'rm': 5.20},
    'J0323+0534': {'flux': 2.766, 'frac_pol': 0.0026, 'pol_angle': -71.6, 'rm': 2.40},
    'J0329+2756': {'flux': 1.395, 'frac_pol': 0.0016, 'pol_angle': -3.7, 'rm': 2.40},
    'J0403+2600': {'flux': 1.289, 'frac_pol': 0.0305, 'pol_angle': 75.3, 'rm': 49.80},
    'J0405-1308': {'flux': 3.945, 'frac_pol': 0.0114, 'pol_angle': 31.4, 'rm': 19.60},
    'J0408-6545': {'flux': 15.198, 'frac_pol': 0.0001, 'pol_angle': -65.1, 'rm': 1.20},
    'J0409-1757': {'flux': 2.205, 'frac_pol': 0.0015, 'pol_angle': -35.3, 'rm': 5.20},
    'J0420-6223': {'flux': 3.327, 'frac_pol': 0.0012, 'pol_angle': 65.7, 'rm': 2.60},
    'J0423-0120': {'flux': 1.195, 'frac_pol': 0.0407, 'pol_angle': 39.0, 'rm': -20.60},
    'J0440-4333': {'flux': 3.593, 'frac_pol': 0.0060, 'pol_angle': -76.6, 'rm': 8.40},
    'J0447-2203': {'flux': 2.025, 'frac_pol': 0.0023, 'pol_angle': -24.6, 'rm': 2.00},
    'J0453-2807': {'flux': 2.193, 'frac_pol': 0.0059, 'pol_angle': 17.0, 'rm': 19.60},
    'J0503+0203': {'flux': 2.218, 'frac_pol': 0.0004, 'pol_angle': 41.6, 'rm': 1.60},
    'J0521+1638': {'flux': 8.332, 'frac_pol': 0.0771, 'pol_angle': -8.2, 'rm': -0.80},
    'J0534+1927': {'flux': 6.741, 'frac_pol': 0.0002, 'pol_angle': -4.0, 'rm': 6.40},
    'J0538-4405': {'flux': 2.156, 'frac_pol': 0.0007, 'pol_angle': 70.8, 'rm': 48.80},
    'J0609-1542': {'flux': 1.674, 'frac_pol': 0.0228, 'pol_angle': 9.2, 'rm': 68.40},
    'J0616-3456': {'flux': 2.904, 'frac_pol': 0.0013, 'pol_angle': -54.1, 'rm': 1.40},
    'J0632+1022': {'flux': 2.427, 'frac_pol': 0.0024, 'pol_angle': 37.0, 'rm': 2.60},
    'J0725-0054': {'flux': 11.733, 'frac_pol': 0.0252, 'pol_angle': -13.2, 'rm': 48.80},
    'J0730-1141': {'flux': 2.260, 'frac_pol': 0.0098, 'pol_angle': -72.9, 'rm': 108.00},
    'J0735-1735': {'flux': 2.600, 'frac_pol': 0.0006, 'pol_angle': 50.0, 'rm': 0.00},
    'J0739+0137': {'flux': 1.021, 'frac_pol': 0.0524, 'pol_angle': -81.9, 'rm': 27.60},
    'J0745+1011': {'flux': 3.225, 'frac_pol': 0.0022, 'pol_angle': 46.4, 'rm': 3.20},
    'J0825-5010': {'flux': 6.244, 'frac_pol': 0.0023, 'pol_angle': 61.3, 'rm': 1.40},
    'J0828-3731': {'flux': 2.087, 'frac_pol': 0.0035, 'pol_angle': -83.9, 'rm': 1.60},
    'J0842+1835': {'flux': 1.039, 'frac_pol': 0.0309, 'pol_angle': 13.9, 'rm': 32.20},
    'J0854+2006': {'flux': 2.047, 'frac_pol': 0.0683, 'pol_angle': 29.8, 'rm': 29.80},
    'J0906-6829': {'flux': 1.818, 'frac_pol': 0.0227, 'pol_angle': -27.0, 'rm': -48.00},
    'J1008+0730': {'flux': 6.533, 'frac_pol': 0.0022, 'pol_angle': -5.9, 'rm': 0.20},
    'J1051-2023': {'flux': 1.442, 'frac_pol': 0.0230, 'pol_angle': 66.7, 'rm': -4.00},
    'J1058+0133': {'flux': 3.672, 'frac_pol': 0.0407, 'pol_angle': -11.5, 'rm': -39.80},
    'J1120-2508': {'flux': 1.638, 'frac_pol': 0.0146, 'pol_angle': -32.1, 'rm': 9.00},
    'J1130-1449': {'flux': 4.838, 'frac_pol': 0.0485, 'pol_angle': 71.2, 'rm': 36.20},
    'J1154-3505': {'flux': 6.084, 'frac_pol': 0.0016, 'pol_angle': 49.1, 'rm': -4.80},
    'J1215-1731': {'flux': 1.700, 'frac_pol': 0.0201, 'pol_angle': 11.9, 'rm': -14.60},
    'J1239-1023': {'flux': 1.554, 'frac_pol': 0.0251, 'pol_angle': 78.8, 'rm': 2.40},
    'J1246-2547': {'flux': 0.854, 'frac_pol': 0.0348, 'pol_angle': -25.4, 'rm': -28.20},
    'J1256-0547': {'flux': 9.782, 'frac_pol': 0.0361, 'pol_angle': -39.8, 'rm': 17.80},
    'J1311-2216': {'flux': 4.857, 'frac_pol': 0.0005, 'pol_angle': 18.5, 'rm': -12.60},
    'J1318-4620': {'flux': 2.205, 'frac_pol': 0.0004, 'pol_angle': -18.1, 'rm': 1.60},
    'J1323-4452': {'flux': 3.026, 'frac_pol': 0.0011, 'pol_angle': 2.8, 'rm': 0.80},
    'J1331+3030': {'flux': 14.259, 'frac_pol': 0.0930, 'pol_angle': 33.1, 'rm': 0.00},
    'J1337-1257': {'flux': 2.525, 'frac_pol': 0.0101, 'pol_angle': 3.5, 'rm': -17.20},
    'J1347+1217': {'flux': 5.213, 'frac_pol': 0.0004, 'pol_angle': 60.0, 'rm': 8.00},
    'J1424-4913': {'flux': 8.131, 'frac_pol': 0.0184, 'pol_angle': -4.4, 'rm': 10.80},
    'J1427-4206': {'flux': 4.464, 'frac_pol': 0.0200, 'pol_angle': -46.1, 'rm': -39.60},
    'J1445+0958': {'flux': 2.166, 'frac_pol': 0.0086, 'pol_angle': -75.7, 'rm': 14.00},
    'J1501-3918': {'flux': 2.806, 'frac_pol': 0.0019, 'pol_angle': 18.4, 'rm': 0.80},
    'J1512-0906': {'flux': 2.442, 'frac_pol': 0.0324, 'pol_angle': 42.0, 'rm': -10.40},
    'J1517-2422': {'flux': 3.026, 'frac_pol': 0.0335, 'pol_angle': 41.0, 'rm': -5.40},
    'J1550+0527': {'flux': 2.864, 'frac_pol': 0.0177, 'pol_angle': 78.1, 'rm': -6.40},
    'J1605-1734': {'flux': 1.376, 'frac_pol': 0.0233, 'pol_angle': -49.1, 'rm': -110.20},
    'J1609+2641': {'flux': 4.619, 'frac_pol': 0.0024, 'pol_angle': -47.9, 'rm': 4.60},
    'J1619-8418': {'flux': 1.514, 'frac_pol': 0.0004, 'pol_angle': 46.6, 'rm': 3.20},
    'J1726-5529': {'flux': 5.208, 'frac_pol': 0.0013, 'pol_angle': -65.5, 'rm': 0.20},
    'J1733-1304': {'flux': 6.215, 'frac_pol': 0.0358, 'pol_angle': -79.7, 'rm': -60.40},
    'J1744-5144': {'flux': 6.917, 'frac_pol': 0.0010, 'pol_angle': -8.7, 'rm': -10.00},
    'J1830-3602': {'flux': 7.227, 'frac_pol': 0.0030, 'pol_angle': -80.4, 'rm': 0.40},
    'J1833-2103': {'flux': 10.665, 'frac_pol': 0.0014, 'pol_angle': -24.9, 'rm': -8.40},
    'J1859-6615': {'flux': 1.602, 'frac_pol': 0.0016, 'pol_angle': 9.2, 'rm': 8.20},
    'J1911-2006': {'flux': 2.284, 'frac_pol': 0.0179, 'pol_angle': -78.4, 'rm': -80.40},
    'J1923-2104': {'flux': 1.215, 'frac_pol': 0.0293, 'pol_angle': -33.1, 'rm': 9.80},
    'J1924-2914': {'flux': 4.937, 'frac_pol': 0.0131, 'pol_angle': 3.4, 'rm': -18.60},
    'J1939-6342': {'flux': 14.554, 'frac_pol': 0.0016, 'pol_angle': -0.2, 'rm': -1.40},
    'J1951-2737': {'flux': 1.297, 'frac_pol': 0.0140, 'pol_angle': -80.2, 'rm': -1.20},
    'J2007-1016': {'flux': 1.504, 'frac_pol': 0.0518, 'pol_angle': -12.4, 'rm': -81.60},
    'J2011-0644': {'flux': 2.628, 'frac_pol': 0.0025, 'pol_angle': -32.0, 'rm': 1.00},
    'J2052-3640': {'flux': 1.367, 'frac_pol': 0.0019, 'pol_angle': -79.3, 'rm': 2.40},
    'J2130+0502': {'flux': 4.033, 'frac_pol': 0.0007, 'pol_angle': 20.8, 'rm': 7.00},
    'J2131-1207': {'flux': 1.970, 'frac_pol': 0.0175, 'pol_angle': -44.1, 'rm': 6.80},
    'J2131-2036': {'flux': 1.961, 'frac_pol': 0.0005, 'pol_angle': -3.0, 'rm': 19.40},
    'J2134-0153': {'flux': 1.852, 'frac_pol': 0.0391, 'pol_angle': 48.0, 'rm': 48.80},
    'J2136+0041': {'flux': 3.867, 'frac_pol': 0.0033, 'pol_angle': -1.6, 'rm': 0.00},
    'J2148+0657': {'flux': 3.274, 'frac_pol': 0.0072, 'pol_angle': -12.1, 'rm': 7.00},
    'J2152-2828': {'flux': 2.918, 'frac_pol': 0.0289, 'pol_angle': 37.0, 'rm': -40.40},
    'J2158-1501': {'flux': 4.053, 'frac_pol': 0.0461, 'pol_angle': -52.8, 'rm': 14.80},
    'J2206-1835': {'flux': 6.284, 'frac_pol': 0.0015, 'pol_angle': -42.4, 'rm': 16.80},
    'J2212+0152': {'flux': 2.915, 'frac_pol': 0.0049, 'pol_angle': -5.8, 'rm': 2.00},
    'J2214-3835': {'flux': 1.810, 'frac_pol': 0.0057, 'pol_angle': -78.8, 'rm': 2.20},
    'J2225-0457': {'flux': 7.717, 'frac_pol': 0.0389, 'pol_angle': -60.9, 'rm': -27.20},
    'J2229-3823': {'flux': 2.034, 'frac_pol': 0.0046, 'pol_angle': -86.4, 'rm': 6.40},
    'J2232+1143': {'flux': 6.939, 'frac_pol': 0.0302, 'pol_angle': -81.7, 'rm': -53.40},
    'J2236+2828': {'flux': 1.795, 'frac_pol': 0.0046, 'pol_angle': -39.6, 'rm': -125.60},
    'J2246-1206': {'flux': 1.791, 'frac_pol': 0.0156, 'pol_angle': 72.2, 'rm': -16.40},
    'J2253+1608': {'flux': 16.199, 'frac_pol': 0.0611, 'pol_angle': 62.8, 'rm': -55.00},
}

def linfit(xInput, xDataList, yDataList):
    """
    Linear interpolation/extrapolation for polarization parameters across frequency.
    """
    y_predict = np.poly1d(np.polyfit(xDataList, yDataList, 1))
    yPredict = y_predict(xInput)
    return yPredict


def get_calibrator_params(source_name):
    """
    Retrieve pol parameters for a given source from hardcoded L-band catalogue.
    Supports multiple naming conventions (e.g., J0240-2309, 0240-2309, 0240).
    """
    if source_name in L_BAND_CALIBRATORS:
        return L_BAND_CALIBRATORS[source_name]

    source_base = source_name.lstrip('J').replace('+', '').replace('-', '')
    for cat_source, params in L_BAND_CALIBRATORS.items():
        cat_base = cat_source.lstrip('J').replace('+', '').replace('-', '')
        if source_base == cat_base:
            return params

    return None


def do_setjy(visname, spw, fields, standard, dopol=False, createmms=True, polcalfield=None):

    delmod(vis=visname)

    fluxlist = ["J0408-6545", "0408-6545", ""]
    ismms = createmms

    msmd.open(visname)
    fnames = fields.fluxfield.split(",")
    for fname in fnames:
        if fname.isdigit():
            fname = msmd.namesforfields(int(fname))

    do_manual = False
    for ff in fluxlist:
        if ff in fnames:
            setjyname = ff
            do_manual = True
            break
        else:
            setjyname = fields.fluxfield.split(",")[0]

    if do_manual:
        smodel = [17.066, 0.0, 0.0, 0.0]
        spix = [-1.179]
        reffreq = "1284MHz"

        logger.info("Using manual flux density scale - ")
        logger.info("Flux model: %s ", smodel)
        logger.info("Spix: %s", spix)
        logger.info("Ref freq %s", reffreq)

        setjy(vis=visname,field=setjyname,scalebychan=True,standard="manual",fluxdensity=smodel,spix=spix,reffreq=reffreq,ismms=ismms)
    else:
        setjy(vis=visname, field=setjyname, spw=spw, scalebychan=True, standard=standard,ismms=ismms)

    fieldnames = msmd.fieldnames()

    if dopol:
        is3C286 = False
        try:
            calibrator_3C286 = list(set(["3C286", "1328+307", "1331+305", "J1331+3030"]).intersection(set(fieldnames)))[0]
        except IndexError:
            calibrator_3C286 = []

        if len(calibrator_3C286):
            is3C286 = True
            id3C286 = str(msmd.fieldsforname(calibrator_3C286)[0])

        if is3C286:
            logger.info("Detected calibrator name(s):  %s" % calibrator_3C286)
            logger.info("Flux and spectral index taken/calculated from:  https://science.nrao.gov/facilities/vla/docs/manuals/oss/performance/fdscale")
            logger.info("Estimating polarization index and position angle of polarized emission from linear fit based on: Perley & Butler 2013 (https://ui.adsabs.harvard.edu/abs/2013ApJS..204...19P/abstract)")
            spwMeanFreq = msmd.meanfreq(0, unit='GHz')
            freqList = np.array([1.05, 1.45, 1.64, 1.95])
            fracPolList = [0.086, 0.095, 0.099, 0.101]
            polindex = linfit(spwMeanFreq, freqList, fracPolList)
            logger.info("Predicted polindex at frequency %s: %s", spwMeanFreq, polindex)
            polPositionAngleList = [33, 33, 33, 33]
            polangle = linfit(spwMeanFreq, freqList, polPositionAngleList)
            logger.info("Predicted pol angle at frequency %s: %s", spwMeanFreq, polangle)

            reffreq = "1.45GHz"
            logger.info("Ref freq %s", reffreq)
            setjy(vis=visname,
                field=id3C286,
                scalebychan=True,
                standard="manual",
                fluxdensity=[-14.6, 0.0, 0.0, 0.0],
                reffreq=reffreq,
                polindex=[polindex],
                polangle=[polangle],
                rotmeas=0,ismms=ismms)

        is3C138 = False
        try:
            calibrator_3C138 = list(set(["3C138", "0518+165", "0521+166", "J0521+1638"]).intersection(set(fieldnames)))[0]
        except IndexError:
            calibrator_3C138 = []

        if len(calibrator_3C138):
            is3C138 = True
            id3C138 = str(msmd.fieldsforname(calibrator_3C138)[0])

        if is3C138:
            logger.info("Detected calibrator name(s):  %s" % calibrator_3C138)
            logger.info("Flux and spectral index taken/calculated from:  https://science.nrao.gov/facilities/vla/docs/manuals/oss/performance/fdscale")
            logger.info("Estimating polarization index and position angle of polarized emission from linear fit based on: Perley & Butler 2013 (https://ui.adsabs.harvard.edu/abs/2013ApJS..204...19P/abstract)")
            spwMeanFreq = msmd.meanfreq(0, unit='GHz')
            freqList = np.array([1.05, 1.45, 1.64, 1.95])
            fracPolList = [0.056, 0.075, 0.084, 0.09]
            polindex = linfit(spwMeanFreq, freqList, fracPolList)
            logger.info("Predicted polindex at frequency %s: %s", spwMeanFreq, polindex)
            polPositionAngleList = [-14, -11, -10, -10]
            polangle = linfit(spwMeanFreq, freqList, polPositionAngleList)
            logger.info("Predicted pol angle at frequency %s: %s", spwMeanFreq, polangle)

            reffreq = "1.45GHz"
            logger.info("Ref freq %s", reffreq)
            setjy(vis=visname,
                field=id3C138,
                scalebychan=True,
                standard="manual",
                fluxdensity=[-8.26, 0.0, 0.0, 0.0],
                reffreq=reffreq,
                polindex=[polindex],
                polangle=[polangle],
                rotmeas=0,ismms=ismms)

        if not (is3C286 or is3C138) and polcalfield:
            polcal_names = [f.strip() for f in polcalfield.split(',')]
            detected_polcal = None

            for polcal in polcal_names:
                if polcal in fieldnames:
                    detected_polcal = polcal
                    break

            if not detected_polcal:
                for polcal in polcal_names:
                    for fieldname in fieldnames:
                        if get_calibrator_params(fieldname):
                            if (fieldname.lstrip('J').replace('+', '').replace('-', '') ==
                                polcal.lstrip('J').replace('+', '').replace('-', '')):
                                detected_polcal = fieldname
                                break
                    if detected_polcal:
                        break

            if detected_polcal:
                pol_params = get_calibrator_params(detected_polcal)
                if pol_params:
                    idPolCal = str(msmd.fieldsforname(detected_polcal)[0])

                    logger.info("Detected L-band calibrator:  %s" % detected_polcal)
                    logger.info("Using L-band polarization parameters from calibrator catalogue")

                    flux_1p4 = pol_params['flux']
                    frac_pol = pol_params['frac_pol']
                    pol_angle = pol_params['pol_angle']
                    rm = pol_params['rm']

                    logger.info("Flux at 1.4 GHz: %s Jy", flux_1p4)
                    logger.info("Fractional polarization at 1.4 GHz: %s", frac_pol)
                    logger.info("Polarization angle at 1.4 GHz: %s degrees", pol_angle)
                    logger.info("Rotation measure: %s rad/m^2", rm)

                    reffreq = "1.4GHz"
                    logger.info("Ref freq %s", reffreq)
                    setjy(vis=visname,
                        field=idPolCal,
                        scalebychan=True,
                        standard="manual",
                        fluxdensity=[flux_1p4, 0.0, 0.0, 0.0],
                        reffreq=reffreq,
                        polindex=[frac_pol],
                        polangle=[pol_angle],
                        rotmeas=rm,ismms=ismms)

    msmd.done()


def main(args,taskvals):

    visname = va(taskvals, "data", "vis", str)

    if os.path.exists(os.path.join(os.getcwd(), "caltables")):
        shutil.rmtree(os.path.join(os.getcwd(), "caltables"))

    calfiles, caldir = bookkeeping.bookkeeping(visname)
    fields = bookkeeping.get_field_ids(taskvals["fields"])

    spw = va(taskvals, "crosscal", "spw", str, default="")
    standard = va(taskvals, "crosscal", "standard", str, default="Stevens-Reynolds 2016")
    dopol = va(taskvals, 'run', 'dopol', bool, default=False)
    createmms = va(taskvals, 'crosscal', 'createmms', bool, default=True)
    polcalfield = va(taskvals, "crosscal", "polcalfield", str, default="")

    do_setjy(visname, spw, fields, standard, dopol, createmms, polcalfield)

if __name__ == '__main__':

    bookkeeping.run_script(main,logfile)
