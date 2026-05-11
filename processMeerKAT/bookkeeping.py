#Copyright (C) 2022 Inter-University Institute for Data Intensive Astronomy
#See processMeerKAT.py for license details.

#!/usr/bin/env python3

import sys
import traceback

import config_parser
from collections import namedtuple
import os
import glob
import re

import logging
from time import gmtime
logging.Formatter.converter = gmtime
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)-15s %(levelname)s: %(message)s", level=logging.INFO)

L_BAND_CALIBRATORS = {
    'J0010-4153': {'flux':  4.545, 'frac_pol': 0.0012, 'pol_angle': -60.2, 'rm':   -1.20},
    'J0022+0014': {'flux':  2.900, 'frac_pol': 0.0009, 'pol_angle': -11.3, 'rm':    2.00},
    'J0024-4202': {'flux':  2.871, 'frac_pol': 0.0010, 'pol_angle':  -4.1, 'rm':    4.60},
    'J0025-2602': {'flux':  8.731, 'frac_pol': 0.0017, 'pol_angle':  72.8, 'rm':   -3.60},
    'J0059+0006': {'flux':  2.449, 'frac_pol': 0.0376, 'pol_angle':  74.3, 'rm':   -3.60},
    'J0108+0134': {'flux':  3.113, 'frac_pol': 0.0388, 'pol_angle': -79.7, 'rm':   -6.60},
    'J0137+3309': {'flux': 16.112, 'frac_pol': 0.0064, 'pol_angle': -36.4, 'rm':  -55.40},
    'J0155-4048': {'flux':  2.161, 'frac_pol': 0.0007, 'pol_angle':  29.9, 'rm':    0.00},
    'J0203-4349': {'flux':  2.726, 'frac_pol': 0.0091, 'pol_angle': -55.5, 'rm':   -8.20},
    'J0210-5101': {'flux':  3.402, 'frac_pol': 0.0124, 'pol_angle':  30.7, 'rm':   14.60},
    'J0238+1636': {'flux':  0.528, 'frac_pol': 0.0077, 'pol_angle':  19.6, 'rm':   48.00},
    'J0240-2309': {'flux':  5.938, 'frac_pol': 0.0098, 'pol_angle': -42.2, 'rm':   10.00},
    'J0252-7104': {'flux':  5.811, 'frac_pol': 0.0018, 'pol_angle': -35.3, 'rm':    2.40},
    'J0303-6211': {'flux':  3.194, 'frac_pol': 0.0471, 'pol_angle':  46.6, 'rm':   49.60},
    'J0318+1628': {'flux':  7.619, 'frac_pol': 0.0003, 'pol_angle': -19.9, 'rm':    5.20},
    'J0323+0534': {'flux':  2.766, 'frac_pol': 0.0026, 'pol_angle': -71.6, 'rm':    2.40},
    'J0329+2756': {'flux':  1.395, 'frac_pol': 0.0016, 'pol_angle':  -3.7, 'rm':    2.40},
    'J0403+2600': {'flux':  1.289, 'frac_pol': 0.0305, 'pol_angle':  75.3, 'rm':   49.80},
    'J0405-1308': {'flux':  3.945, 'frac_pol': 0.0114, 'pol_angle':  31.4, 'rm':   19.60},
    'J0408-6545': {'flux': 15.198, 'frac_pol': 0.0001, 'pol_angle': -65.1, 'rm':    1.20},
    'J0409-1757': {'flux':  2.205, 'frac_pol': 0.0015, 'pol_angle': -35.3, 'rm':    5.20},
    'J0420-6223': {'flux':  3.327, 'frac_pol': 0.0012, 'pol_angle':  65.7, 'rm':    2.60},
    'J0423-0120': {'flux':  1.195, 'frac_pol': 0.0407, 'pol_angle':  39.0, 'rm':  -20.60},
    'J0440-4333': {'flux':  3.593, 'frac_pol': 0.0060, 'pol_angle': -76.6, 'rm':    8.40},
    'J0447-2203': {'flux':  2.025, 'frac_pol': 0.0023, 'pol_angle': -24.6, 'rm':    2.00},
    'J0453-2807': {'flux':  2.193, 'frac_pol': 0.0059, 'pol_angle':  17.0, 'rm':   19.60},
    'J0503+0203': {'flux':  2.218, 'frac_pol': 0.0004, 'pol_angle':  41.6, 'rm':    1.60},
    'J0521+1638': {'flux':  8.332, 'frac_pol': 0.0771, 'pol_angle':  -8.2, 'rm':   -0.80},
    'J0534+1927': {'flux':  6.741, 'frac_pol': 0.0002, 'pol_angle':  -4.0, 'rm':    6.40},
    'J0538-4405': {'flux':  2.156, 'frac_pol': 0.0007, 'pol_angle':  70.8, 'rm':   48.80},
    'J0609-1542': {'flux':  1.674, 'frac_pol': 0.0228, 'pol_angle':   9.2, 'rm':   68.40},
    'J0616-3456': {'flux':  2.904, 'frac_pol': 0.0013, 'pol_angle': -54.1, 'rm':    1.40},
    'J0632+1022': {'flux':  2.427, 'frac_pol': 0.0024, 'pol_angle':  37.0, 'rm':    2.60},
    'J0725-0054': {'flux': 11.733, 'frac_pol': 0.0252, 'pol_angle': -13.2, 'rm':   48.80},
    'J0730-1141': {'flux':  2.260, 'frac_pol': 0.0098, 'pol_angle': -72.9, 'rm':  108.00},
    'J0735-1735': {'flux':  2.600, 'frac_pol': 0.0006, 'pol_angle':  50.0, 'rm':    0.00},
    'J0739+0137': {'flux':  1.021, 'frac_pol': 0.0524, 'pol_angle': -81.9, 'rm':   27.60},
    'J0745+1011': {'flux':  3.225, 'frac_pol': 0.0022, 'pol_angle':  46.4, 'rm':    3.20},
    'J0825-5010': {'flux':  6.244, 'frac_pol': 0.0023, 'pol_angle':  61.3, 'rm':    1.40},
    'J0828-3731': {'flux':  2.087, 'frac_pol': 0.0035, 'pol_angle': -83.9, 'rm':    1.60},
    'J0842+1835': {'flux':  1.039, 'frac_pol': 0.0309, 'pol_angle':  13.9, 'rm':   32.20},
    'J0854+2006': {'flux':  2.047, 'frac_pol': 0.0683, 'pol_angle':  29.8, 'rm':   29.80},
    'J0906-6829': {'flux':  1.818, 'frac_pol': 0.0227, 'pol_angle': -27.0, 'rm':  -48.00},
    'J1008+0730': {'flux':  6.533, 'frac_pol': 0.0022, 'pol_angle':  -5.9, 'rm':    0.20},
    'J1051-2023': {'flux':  1.442, 'frac_pol': 0.0230, 'pol_angle':  66.7, 'rm':   -4.00},
    'J1058+0133': {'flux':  3.672, 'frac_pol': 0.0407, 'pol_angle': -11.5, 'rm':  -39.80},
    'J1120-2508': {'flux':  1.638, 'frac_pol': 0.0146, 'pol_angle': -32.1, 'rm':    9.00},
    'J1130-1449': {'flux':  4.838, 'frac_pol': 0.0485, 'pol_angle':  71.2, 'rm':   36.20},
    'J1154-3505': {'flux':  6.084, 'frac_pol': 0.0016, 'pol_angle':  49.1, 'rm':   -4.80},
    'J1215-1731': {'flux':  1.700, 'frac_pol': 0.0201, 'pol_angle':  11.9, 'rm':  -14.60},
    'J1239-1023': {'flux':  1.554, 'frac_pol': 0.0251, 'pol_angle':  78.8, 'rm':    2.40},
    'J1246-2547': {'flux':  0.854, 'frac_pol': 0.0348, 'pol_angle': -25.4, 'rm':  -28.20},
    'J1256-0547': {'flux':  9.782, 'frac_pol': 0.0361, 'pol_angle': -39.8, 'rm':   17.80},
    'J1311-2216': {'flux':  4.857, 'frac_pol': 0.0005, 'pol_angle':  18.5, 'rm':  -12.60},
    'J1318-4620': {'flux':  2.205, 'frac_pol': 0.0004, 'pol_angle': -18.1, 'rm':    1.60},
    'J1323-4452': {'flux':  3.026, 'frac_pol': 0.0011, 'pol_angle':   2.8, 'rm':    0.80},
    'J1331+3030': {'flux': 14.259, 'frac_pol': 0.0930, 'pol_angle':  33.1, 'rm':    0.00},
    'J1337-1257': {'flux':  2.525, 'frac_pol': 0.0101, 'pol_angle':   3.5, 'rm':  -17.20},
    'J1347+1217': {'flux':  5.213, 'frac_pol': 0.0004, 'pol_angle':  60.0, 'rm':    8.00},
    'J1424-4913': {'flux':  8.131, 'frac_pol': 0.0184, 'pol_angle':  -4.4, 'rm':   10.80},
    'J1427-4206': {'flux':  4.464, 'frac_pol': 0.0200, 'pol_angle': -46.1, 'rm':  -39.60},
    'J1445+0958': {'flux':  2.166, 'frac_pol': 0.0086, 'pol_angle': -75.7, 'rm':   14.00},
    'J1501-3918': {'flux':  2.806, 'frac_pol': 0.0019, 'pol_angle':  18.4, 'rm':    0.80},
    'J1512-0906': {'flux':  2.442, 'frac_pol': 0.0324, 'pol_angle':  42.0, 'rm':  -10.40},
    'J1517-2422': {'flux':  3.026, 'frac_pol': 0.0335, 'pol_angle':  41.0, 'rm':   -5.40},
    'J1550+0527': {'flux':  2.864, 'frac_pol': 0.0177, 'pol_angle':  78.1, 'rm':   -6.40},
    'J1605-1734': {'flux':  1.376, 'frac_pol': 0.0233, 'pol_angle': -49.1, 'rm': -110.20},
    'J1609+2641': {'flux':  4.619, 'frac_pol': 0.0024, 'pol_angle': -47.9, 'rm':    4.60},
    'J1619-8418': {'flux':  1.514, 'frac_pol': 0.0004, 'pol_angle':  46.6, 'rm':    3.20},
    'J1726-5529': {'flux':  5.208, 'frac_pol': 0.0013, 'pol_angle': -65.5, 'rm':    0.20},
    'J1733-1304': {'flux':  6.215, 'frac_pol': 0.0358, 'pol_angle': -79.7, 'rm':  -60.40},
    'J1744-5144': {'flux':  6.917, 'frac_pol': 0.0010, 'pol_angle':  -8.7, 'rm':  -10.00},
    'J1830-3602': {'flux':  7.227, 'frac_pol': 0.0030, 'pol_angle': -80.4, 'rm':    0.40},
    'J1833-2103': {'flux': 10.665, 'frac_pol': 0.0014, 'pol_angle': -24.9, 'rm':   -8.40},
    'J1859-6615': {'flux':  1.602, 'frac_pol': 0.0016, 'pol_angle':   9.2, 'rm':    8.20},
    'J1911-2006': {'flux':  2.284, 'frac_pol': 0.0179, 'pol_angle': -78.4, 'rm':  -80.40},
    'J1923-2104': {'flux':  1.215, 'frac_pol': 0.0293, 'pol_angle': -33.1, 'rm':    9.80},
    'J1924-2914': {'flux':  4.937, 'frac_pol': 0.0131, 'pol_angle':   3.4, 'rm':  -18.60},
    'J1939-6342': {'flux': 14.554, 'frac_pol': 0.0016, 'pol_angle':  -0.2, 'rm':   -1.40},
    'J1951-2737': {'flux':  1.297, 'frac_pol': 0.0140, 'pol_angle': -80.2, 'rm':   -1.20},
    'J2007-1016': {'flux':  1.504, 'frac_pol': 0.0518, 'pol_angle': -12.4, 'rm':  -81.60},
    'J2011-0644': {'flux':  2.628, 'frac_pol': 0.0025, 'pol_angle': -32.0, 'rm':    1.00},
    'J2052-3640': {'flux':  1.367, 'frac_pol': 0.0019, 'pol_angle': -79.3, 'rm':    2.40},
    'J2130+0502': {'flux':  4.033, 'frac_pol': 0.0007, 'pol_angle':  20.8, 'rm':    7.00},
    'J2131-1207': {'flux':  1.970, 'frac_pol': 0.0175, 'pol_angle': -44.1, 'rm':    6.80},
    'J2131-2036': {'flux':  1.961, 'frac_pol': 0.0005, 'pol_angle':  -3.0, 'rm':   19.40},
    'J2134-0153': {'flux':  1.852, 'frac_pol': 0.0391, 'pol_angle':  48.0, 'rm':   48.80},
    'J2136+0041': {'flux':  3.867, 'frac_pol': 0.0033, 'pol_angle':  -1.6, 'rm':    0.00},
    'J2148+0657': {'flux':  3.274, 'frac_pol': 0.0072, 'pol_angle': -12.1, 'rm':    7.00},
    'J2152-2828': {'flux':  2.918, 'frac_pol': 0.0289, 'pol_angle':  37.0, 'rm':  -40.40},
    'J2158-1501': {'flux':  4.053, 'frac_pol': 0.0461, 'pol_angle': -52.8, 'rm':   14.80},
    'J2206-1835': {'flux':  6.284, 'frac_pol': 0.0015, 'pol_angle': -42.4, 'rm':   16.80},
    'J2212+0152': {'flux':  2.915, 'frac_pol': 0.0049, 'pol_angle':  -5.8, 'rm':    2.00},
    'J2214-3835': {'flux':  1.810, 'frac_pol': 0.0057, 'pol_angle': -78.8, 'rm':    2.20},
    'J2225-0457': {'flux':  7.717, 'frac_pol': 0.0389, 'pol_angle': -60.9, 'rm':  -27.20},
    'J2229-3823': {'flux':  2.034, 'frac_pol': 0.0046, 'pol_angle': -86.4, 'rm':    6.40},
    'J2232+1143': {'flux':  6.939, 'frac_pol': 0.0302, 'pol_angle': -81.7, 'rm':  -53.40},
    'J2236+2828': {'flux':  1.795, 'frac_pol': 0.0046, 'pol_angle': -39.6, 'rm': -125.60},
    'J2246-1206': {'flux':  1.791, 'frac_pol': 0.0156, 'pol_angle':  72.2, 'rm':  -16.40},
    'J2253+1608': {'flux': 16.199, 'frac_pol': 0.0611, 'pol_angle':  62.8, 'rm':  -55.00},
}


def get_calibrator_params(source_name):
    """Return L-band pol parameters for source_name, or None if not in catalogue."""
    if source_name in L_BAND_CALIBRATORS:
        return L_BAND_CALIBRATORS[source_name]
    source_base = source_name.lstrip('J').replace('+', '').replace('-', '')
    for cat_source, params in L_BAND_CALIBRATORS.items():
        cat_base = cat_source.lstrip('J').replace('+', '').replace('-', '')
        if source_base == cat_base:
            return params
    return None


def get_calfiles(visname, caldir):
        base = os.path.splitext(visname)[0]
        kcorrfile = os.path.join(caldir,base + '.kcal')
        bpassfile = os.path.join(caldir,base + '.bcal')
        gainfile =  os.path.join(caldir,base + '.gcal')
        dpolfile =  os.path.join(caldir,base + '.pcal')
        xpolfile =  os.path.join(caldir,base + '.xcal')
        xdelfile =  os.path.join(caldir,base + '.xdel')
        fluxfile =  os.path.join(caldir,base + '.fluxscale')

        calfiles = namedtuple('calfiles',
                ['kcorrfile', 'bpassfile', 'gainfile', 'dpolfile', 'xpolfile',
                    'xdelfile', 'fluxfile'])
        return calfiles(kcorrfile, bpassfile, gainfile, dpolfile, xpolfile,
                xdelfile, fluxfile)


def bookkeeping(visname):
    # Book keeping
    caldir = os.path.join(os.getcwd(), 'caltables')
    calfiles = get_calfiles(visname, caldir)

    return calfiles, caldir

def get_field_ids(fields):
    """
    Given an input list of source names, finds the associated field
    IDS from the MS and returns them as a list.
    """

    targetfield    = fields['targetfields']
    extrafields    = fields['extrafields']
    fluxfield      = fields['fluxfield']
    bpassfield     = fields['bpassfield']
    secondaryfield = fields['phasecalfield']
    kcorrfield     = fields['phasecalfield']
    xdelfield      = fields['phasecalfield']
    dpolfield      = fields['phasecalfield']
    xpolfield      = fields['phasecalfield']

    if fluxfield != secondaryfield:
        gainfields = \
                str(fluxfield) + ',' + str(secondaryfield)
    else:
        gainfields = str(fluxfield)

    FieldIDs = namedtuple('FieldIDs', ['targetfield', 'fluxfield',
                    'bpassfield', 'secondaryfield', 'kcorrfield', 'xdelfield',
                    'dpolfield', 'xpolfield', 'gainfields', 'extrafields'])

    return FieldIDs(targetfield, fluxfield, bpassfield, secondaryfield,
            kcorrfield, xdelfield, dpolfield, xpolfield, gainfields, extrafields)

def polfield_name(visname, polcalfield=None):

    from casatools import msmetadata
    msmd = msmetadata()
    msmd.open(visname)
    fieldnames = msmd.fieldnames()
    msmd.done()

    polfield = ''
    if any([ff in ["3C286", "1328+307", "1331+305", "J1331+3030"] for ff in fieldnames]):
        polfield= list(set(["3C286", "1328+307", "1331+305", "J1331+3030"]).intersection(set(fieldnames)))[0]
    elif any([ff in ["3C138", "0518+165", "0521+166", "J0521+1638"] for ff in fieldnames]):
        polfield = list(set(["3C138", "0518+165", "0521+166", "J0521+1638"]).intersection(set(fieldnames)))[0]
    elif any([ff in ["3C48", "0134+329", "0137+331", "J0137+3309"] for ff in fieldnames]):
        polfield = list(set(["3C48", "0134+329", "0137+331", "J0137+3309"]).intersection(set(fieldnames)))[0]
    elif "J1130-1449" in fieldnames:
        polfield = "J1130-1449"
    elif polcalfield:
        for name in [f.strip() for f in polcalfield.split(",")]:
            if name in fieldnames and get_calibrator_params(name) is not None:
                polfield = name
                logger.info("Using '%s' from L-band calibrator catalogue as polarization field." % polfield)
                break
        if polfield == '':
            logger.warning("No valid polarization field found. Defaulting to use the phase calibrator to solve for XY phase.")
            logger.warning("The polarization solutions found will likely be wrong. Please check the results carefully.")
    else:
        logger.warning("No valid polarization field found. Defaulting to use the phase calibrator to solve for XY phase.")
        logger.warning("The polarization solutions found will likely be wrong. Please check the results carefully.")

    return polfield

def check_file(filepath):

    # Python2 only has IOError, so define FileNotFound
    try:
        FileNotFoundError
    except NameError:
        FileNotFoundError = IOError

    if not os.path.exists(filepath):
        logger.error('Calibration table "{0}" was not written. Please check the CASA output and whether a solution was found.'.format(filepath))
        raise FileNotFoundError
    else:
        logger.info('Calibration table "{0}" successfully written.'.format(filepath))

def get_selfcal_params():

    #Flag for input errors
    exit = False

    # Get the name of the config file
    args = config_parser.parse_args()

    # Parse config file
    taskvals, config = config_parser.parse_config(args['config'])
    params = taskvals['selfcal']
    other_params = list(params.keys())

    params['vis'] = taskvals['data']['vis']
    params['refant'] = taskvals['crosscal']['refant']
    params['dopol'] = taskvals['run']['dopol']

    if params['dopol'] and 'G' in params['gaintype']:
        logger.warning("dopol is True, but gaintype includes 'G'. Use gaintype='T' for polarisation on linear feeds (e.g. MeerKAT).")

    single_args = ['nloops','loop','discard_nloops','outlier_threshold','outlier_radius'] #need to be 1 long (i.e. not a list)
    gaincal_args = ['solint','calmode','gaintype','flag'] #need to be nloops long
    list_args = ['imsize'] #allowed to be lists of lists

    for arg in single_args:
        if arg in other_params:
            other_params.pop(other_params.index(arg))

    for arg in single_args:
        if type(params[arg]) is list or type(params[arg]) is str and ',' in params[arg]:
            logger.error("Parameter '{0}' in '{1}' cannot be a list. It must be a single value.".format(arg,args['config']))
            exit = True

    for arg in other_params:
        if type(params[arg]) is str and ',' in params[arg]:
            logger.error("Parameter '{0}' in '{1}' cannot use comma-seprated values. It must be a list or values, or a single value.".format(arg,args['config']))
            exit = True

        # These can be a list of lists or a simple list (if specifying a single value).
        # So make sure these two cases are covered.
        if arg in list_args:
            # Not a list of lists, so turn it into one of right length
            if type(params[arg]) is list and (len(params[arg]) == 0 or type(params[arg][0]) is not list):
                params[arg] = [params[arg],] * (params['nloops'] + 1)
            # Not a list at all, so put it into a list
            elif type(params[arg]) is not list:
                params[arg] = [[params[arg],],] * (params['nloops'] + 1)
            # A list of lists of length 1, so put into list of lists of right length
            elif type(params[arg]) is list and type(params[arg][0]) is list and len(params[arg]) == 1:
                params[arg] = [params[arg][0],] * (params['nloops'] + 1)

        elif type(params[arg]) is not list:
            if arg in gaincal_args:
                params[arg] = [params[arg]] * (params['nloops'])
            else:
                params[arg] = [params[arg]] * (params['nloops'] + 1)

    for arg in other_params:
        #By this point params[arg] will be a list
        if arg in gaincal_args and len(params[arg]) != params['nloops']:
            logger.error("Parameter '{0}' in '{1}' is the wrong length. It is {2} long but must be 'nloops' ({3}) long or a single value (not a list).".format(arg,args['config'],len(params[arg]),params['nloops']))
            exit = True

        elif arg not in gaincal_args and len(params[arg]) != params['nloops'] + 1:
            logger.error("Parameter '{0}' in '{1}' is the wrong length. It is {2} long but must be 'nloops' + 1 ({3}) long or a single value (not a list).".format(arg,args['config'],len(params[arg]),params['nloops']+1))
            exit = True

    if exit:
        sys.exit(1)

    return args,params

def get_selfcal_args(vis,loop,nloops,nterms,deconvolver,discard_nloops,calmode,outlier_threshold,outlier_radius,threshold,step):

    from casatools import msmetadata,quanta
    from read_ms import check_spw
    msmd = msmetadata()
    qa = quanta()

    if os.path.exists('{0}/SUBMSS'.format(vis)):
        tmpvis = glob.glob('{0}/SUBMSS/*'.format(vis))[0]
    else:
        tmpvis = vis

    msmd.open(tmpvis)

    visbase = os.path.split(vis.rstrip('/ '))[1] # Get only vis name, not entire path
    visbase = re.sub('\.\d+\.*\d*\~\d+\.*\d*[a-z,A-Z]?[Hz,hz,hZ,HZ]*\.','.',visbase) # Strip any SPWs from basename (when running outlier imaging separately per SPW)
    targetfields = config_parser.get_key(config_parser.parse_args()['config'], 'fields', 'targetfields')

    #Force taking first target field (relevant for writing outliers.txt at beginning of pipeline)
    if type(targetfields) is str and ',' in targetfields:
        targetfield = targetfields.split(',')[0]
        msg = 'Multiple target fields input ("{0}"), but only one position can be used to identify outliers (for outlier imaging). Using "{1}".'
        logger.warning(msg.format(targetfields,targetfield))
    else:
        targetfield = targetfields
    #Make sure it's an integer
    try:
        targetfield = int(targetfield)
    except ValueError: # It's not an int, but a str
        targetfield = msmd.fieldsforname(targetfield)[0]

    target_str = msmd.namesforfields(targetfield)[0]

    if '.ms' in visbase and target_str not in visbase:
        basename = visbase.replace('.ms','.{0}'.format(target_str))
    else:
        basename = visbase.replace('.mms', '')

    imbase = basename + '_im_%d' # Images will be produced in $CWD
    imagename = imbase % loop
    outimage = imagename + '.image'
    pixmask = imagename + ".pixmask"
    maskfile = imagename + ".islmask"
    rmsfile = imagename + ".rms"
    caltable = basename + '.gcal%d' % loop
    prev_caltables = sorted(glob.glob('*.gcal?'))
    cfcache = basename + '.cf'
    thresh = 10

    if deconvolver[loop] == 'mtmfs':
        outimage += '.tt0'

    if step not in ['tclean','sky'] and not os.path.exists(outimage):
        logger.error("Image '{0}' doesn't exist, so self-calibration loop {1} failed. Will terminate selfcal process.".format(outimage,loop))
        sys.exit(1)

    if step in ['tclean','predict']:
        pixmask = imbase % (loop-1) + '.pixmask'
        rmsfile = imbase % (loop-1) + '.rms'
    if step in ['tclean','predict','sky'] and ((loop == 0 and not os.path.exists(pixmask)) or (0 < loop < nloops and calmode[loop] == '')):
        pixmask = ''

    #Check no missing caltables
    for i in range(0,loop):
        if calmode[i] != '' and not os.path.exists(basename + '.gcal%d' % i):
            logger.error("Calibration table '{0}' doesn't exist, so self-calibration loop {1} failed. Will terminate selfcal process.".format(basename + '.gcal%d' % i,i))
            sys.exit(1)
    for i in range(discard_nloops):
        prev_caltables.pop(0)

    if outlier_threshold != '' and outlier_threshold != 0: # and (loop > 0 or step in ['sky','bdsf'] and loop == 0):
        if step in ['tclean','predict','sky']:
            outlierfile = 'outliers_loop{0}.txt'.format(loop)
        else:
            outlierfile = 'outliers_loop{0}.txt'.format(loop+1)

        #Derive sky model radius for outliers, assuming channel 0 (of SPW 0) is lowest frequency and therefore largest FWHM
        if outlier_radius == 0.0 or outlier_radius == '' and step == 'sky':
            SPW = check_spw(config_parser.parse_args()['config'],msmd)
            low_freq = float(SPW.replace('*:','').split('~')[0]) * 1e6 #MHz to Hz
            rads=1.025*qa.constants(v='c')['value']/low_freq/ msmd.antennadiameter()['0']['value']
            FWHM=qa.convert(qa.quantity(rads,'rad'),'deg')['value']
            sky_model_radius = 1.5*FWHM #degrees
            logger.warning('Using calculated search radius of {0:.1f} degrees.'.format(sky_model_radius))
        else:
            if step == 'sky':
                logger.info('Using preset search radius of {0} degrees'.format(outlier_radius))
            sky_model_radius = outlier_radius
    else:
        outlierfile = ''
        sky_model_radius = 0.0

    msmd.done()

    if not (type(threshold[loop]) is str and 'Jy' in threshold[loop]) and threshold[loop] > 1:
        if step in ['tclean','predict']:
            if os.path.exists(rmsfile):
                from casatasks import imstat
                stats = imstat(imagename=rmsfile)
                threshold[loop] *= stats['min'][0]
            else:
                logger.error("'{0}' doesn't exist. Can't do thresholding at S/N > {1}. Loop 0 must use an absolute threshold value. Check the logs to see why RMS map not created.".format(rmsfile,threshold[loop]))
                sys.exit(1)
        elif step == 'bdsf':
            thresh = threshold[loop]

    return imbase,imagename,outimage,pixmask,rmsfile,caltable,prev_caltables,threshold,outlierfile,cfcache,thresh,maskfile,targetfield,sky_model_radius

def rename_logs(logfile=''):

    if logfile != '' and os.path.exists(logfile):
        if 'SLURM_ARRAY_JOB_ID' in os.environ:
            IDs = '{SLURM_JOB_NAME}-{SLURM_ARRAY_JOB_ID}_{SLURM_ARRAY_TASK_ID}'.format(**os.environ)
        else:
            IDs = '{SLURM_JOB_NAME}-{SLURM_JOB_ID}'.format(**os.environ)

        os.rename(logfile,'logs/{0}.mpi'.format(IDs))
        for log in glob.glob('*.last'):
            os.rename(log,'logs/{0}-{1}.last'.format(os.path.splitext(log)[0],IDs))

def get_imaging_params():

    # Get the name of the config file
    args = config_parser.parse_args()

    # Parse config file
    taskvals, config = config_parser.parse_config(args['config'])
    params = taskvals['image']
    params['vis'] = taskvals['data']['vis']
    params['keepmms'] = taskvals['crosscal']['keepmms']

    #Rename the masks that were already used
    if params['outlierfile'] != '' and os.path.exists(params['outlierfile']):
        outliers=open(params['outlierfile']).read()
        outlier_bases = re.findall(r'imagename=(.*)\n',outliers)
        for name in outlier_bases:
            mask = '{0}.mask'.format(name)
            if os.path.exists(mask):
                newname = '{0}.old'.format(mask)
                logger.info('Re-using old mask for "{0}". Renaming "{1}" to "{2}" to avoid mask conflict.'.format(name,mask,newname))
                os.rename(mask,newname)

    return args,params

def run_script(func,logfile=''):

    # Get the name of the config file
    args = config_parser.parse_args()

    # Parse config file
    taskvals, config = config_parser.parse_config(args['config'])

    continue_run = config_parser.validate_args(taskvals, 'run', 'continue', bool, default=True)
    spw = config_parser.validate_args(taskvals, 'crosscal', 'spw', str)
    nspw = config_parser.validate_args(taskvals, 'crosscal', 'nspw', int)

    if continue_run:
        try:
            func(args,taskvals)
            rename_logs(logfile)
        except Exception as err:
            logger.error('Exception found in the pipeline of type {0}: {1}'.format(type(err),err))
            logger.error(traceback.format_exc())
            config_parser.overwrite_config(args['config'], conf_dict={'continue' : False}, conf_sec='run', sec_comment='# Internal variables for pipeline execution')
            if nspw > 1:
                for SPW in spw.split(','):
                    spw_config = '{0}/{1}'.format(SPW.replace('*:',''),args['config'])
                    config_parser.overwrite_config(spw_config, conf_dict={'continue' : False}, conf_sec='run', sec_comment='# Internal variables for pipeline execution')
            rename_logs(logfile)
            sys.exit(1)
    else:
        logger.error('Exception found in previous pipeline job, which set "continue=False" in [run] section of "{0}". Skipping "{1}".'.format(args['config'],os.path.split(sys.argv[2])[1]))
        #os.system('./killJobs.sh') # and cancelling remaining jobs (scancel not found since /opt overwritten)
        rename_logs(logfile)
        sys.exit(1)
