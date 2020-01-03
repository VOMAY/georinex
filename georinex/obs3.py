from pathlib import Path
import numpy as np
import logging
from datetime import datetime, timedelta
import io
import xarray
from typing import Dict, Union, List, Tuple, Any, Sequence
from typing.io import TextIO
try:
    from pymap3d import ecef2geodetic
except ImportError:
    ecef2geodetic = None
#
from .rio import opener, rinexinfo, HeaderClass
from .common import check_time_interval, check_unique_times
"""https://github.com/mvglasow/satstat/wiki/NMEA-IDs"""

SBAS = 100  # offset for ID
GLONASS = 37
QZSS = 192
BEIDOU = 0


def rinexobs3(fn: Union[TextIO, str, Path],
              use: Sequence[str] = None,
              tlim: Tuple[datetime, datetime] = None,
              useindicators: bool = False,
              meas: Sequence[str] = None,
              verbose: bool = False,
              *,
              fast: bool = False,
              interval: Union[float, int, timedelta] = None) -> xarray.Dataset:
    """
    process RINEX 3 OBS data

    fn: RINEX OBS 3 filename
    use: 'G'  or ['G', 'R'] or similar

    tlim: read between these time bounds
    useindicators: SSI, LLI are output
    meas:  'L1C'  or  ['L1C', 'C1C'] or similar

    fast:
          TODO: FUTURE, not yet enabled for OBS3
          speculative preallocation based on minimum SV assumption and file size.
          Avoids double-reading file and more complicated linked lists.
          Believed that Numpy array should be faster than lists anyway.
          Reduce Nsvmin if error (let us know)

    interval: allows decimating file read by time e.g. every 5 seconds.
                Useful to speed up reading of very large RINEX files
    """

    interval = check_time_interval(interval)

    if isinstance(use, str):
        use = [use]

    if isinstance(meas, str):
        meas = [meas]

    if not use or not use[0].strip():
        use = None

    if not meas or not meas[0].strip():
        meas = None

    if tlim is not None and not isinstance(tlim[0], datetime):
        raise TypeError('time bounds are specified as datetime.datetime')

    last_epoch = None
# %% loop
    with opener(fn) as f:
        try:
            # Read header into HeaderClass instance
            hdr = obsheader3(f)

            # filter signales based on selection via input arguments
            selObs, selInd = filterObs3(hdr, use, meas)

            # Get set of all signals of all constellations
            signalUnion = sorted(set([ signal for constSigList in selObs.values() for signal in constSigList]))
            # Allocate Main internal data buffer
            bufDict = {}
            for signal in signalUnion:
                bufDict[signal] = {'time': [], 'const': [], 'prn': [], 'val': []}

        except KeyError:
            return xarray.Dataset()

# %% process OBS file
        for ln in f:
            if not ln.startswith('>'):  # end of file
                break

            try:
                time, in_range = _timeobs(ln, tlim, last_epoch, interval)
            except ValueError:  # garbage between header and RINEX data
                logging.debug(f'garbage detected in {fn}, trying to parse at next time step')
                continue

            # Number of visible satellites this time %i3  pg. A13
            nSv = int(ln[33:35])
            if in_range == -1:
                for _ in range(nSv):
                    next(f)
                continue

            if in_range == 1:
                break

            last_epoch = time

            if verbose:
                print(time, end="\r")

            # this time epoch is complete, assemble the data.
            obsd = {}
            for _, epochLine in zip(range(nSv), f):
                # Check if this line starts with an expected constellatin letter
                if epochLine[0] not in hdr.obsType.keys():
                    raise KeyError(f'Unexpected line found in RINEX file')

                obsd = _epoch(obsd, selObs, selInd, epochLine)

            for signal in obsd:
                bufDict[signal]['time'].append(time)
                bufDict[signal]['const'].append(obsd[signal]['const'])
                bufDict[signal]['prn'].append(obsd[signal]['prn'])
                bufDict[signal]['val'].append(obsd[signal]['val'])

    data = []
    for signal in bufDict:
        signalTime = bufDict[signal]['time']
        signalConst = np.sort(np.array(list(set([const for constEpochList in bufDict[signal]['const'] for const in constEpochList]))))
        signalPrn = np.sort(np.array(list(set([prn for prnEpochList in bufDict[signal]['prn'] for prn in prnEpochList]))))
        signalVal = np.empty((len(signalTime), len(signalConst), len(signalPrn)))

        data.append(_gen_array(signalTime, signalConst, signalPrn,
                               bufDict[signal]['const'], bufDict[signal]['prn'], bufDict[signal]['val'],
                               np.copy(signalVal), signal))
#        if 'ssi' not in dict_meas[sm]:
#            continue
#        data.append(_gen_array(alltime, allsv, dict_meas[sm]['sv'], dict_meas[sm]['ssi'], np.copy(valarray), sm+'-ssi'))
#        if 'lli' not in dict_meas[sm]:
#            continue
#        data.append(_gen_array(alltime, allsv, dict_meas[sm]['sv'], dict_meas[sm]['lli'], np.copy(valarray), sm+'-lli'))
#

    data = xarray.merge(data)

    # Add Attributes
    data.attrs['version'] = hdr.version
    hdr.cInterval(data.time)
    data.attrs['interval'] = hdr.interval
    data.attrs['rinexType'] = hdr.rinexType
    if hasattr(hdr, 'position'):
        data.attrs['position'] = hdr.position
    if hasattr(hdr, 'positionGeodetic'):
        data.attrs['positionGeodetic'] = hdr.positionGeodetic
    data.attrs['timeSystem'] = hdr.timeSystem
    if isinstance(fn, Path):
        data.attrs['filename'] = fn.name
    if hasattr(hdr, 'tFirst'):
        data.attrs['tFirst'] = hdr.tFirst
    if hasattr(hdr, 'tLast'):
        data.attrs['tLast'] = hdr.tLast

    return data


def _timeobs(ln: str, tlim: Tuple[datetime, datetime] = None,
             last_epoch: datetime = None, interval: timedelta = None) -> Tuple[datetime, int]:
    """
    convert time from RINEX 3 OBS text to datetime
    """

    curr_time = datetime(int(ln[2:6]), int(ln[7:9]), int(ln[10:12]),
                         hour=int(ln[13:15]), minute=int(ln[16:18]),
                         second=int(ln[19:21]),
                         microsecond=int(float(ln[19:29]) % 1 * 1000000))

    in_range = 0
    if tlim is not None:
        if curr_time < tlim[0]:
            in_range = -1
        if curr_time > tlim[1]:
            in_range = 1

    if interval is not None and last_epoch is not None and in_range == 0:
        in_range = -1 if (curr_time - last_epoch < interval) else 0

    return (curr_time, in_range)


def obstime3(fn: Union[TextIO, Path],
             verbose: bool = False) -> np.ndarray:
    """
    return all times in RINEX file
    """
    times = []

    with opener(fn) as f:
        for ln in f:
            if ln.startswith('>'):
                times.append(_timeobs(ln)[0])

    return np.asarray(times)

def _epoch(obsd: Dict[str, Any],
           selObs: Dict[str, Any],
           selInd: Dict[str, Any],
           line: str) -> (Dict[str, Any]):
    """
    processing of each line in epoch (time step)
    """

    # Check if constellation is selected
    sv = line[0:3].replace(' ', '0')
    const = sv[0]
    prn = int(sv[1:3])
    if const not in selObs.keys():
        return obsd

    # Ensure 16-columns per record and filter the selected records into list
    recordStr = line[3:]
    recordStr = recordStr + ' '*(len(recordStr) % 16)
    selRecord = [recordStr[i*16:i*16+16] for i in selInd[const]]

    # Store information in buffer dictionary
    for i, signal in enumerate(selObs[const]):
        # Create key if not available
        if signal not in obsd.keys():
            obsd[signal] = {'const': [], 'prn': [], 'val': []}

        obsd[signal]['const'].append(const)
        obsd[signal]['prn'].append(prn)
        obsd[signal]['val'].append(float(selRecord[i][:14]) if selRecord[i][:14].strip() else np.nan)
#        obsd[signal]['ssi'].(int(selRecord[i][15]) if selRecord[i][15].strip() else np.nan)
#        obsd[signal]['lli'](int(selRecord[i][16]) if selRecord[i][16].strip() else np.nan)

#
#    for (sm, idx) in gen_filter_meas:
#        if idx >= len(parts):
#            continue
#
#        if not parts[idx].strip():
#            continue
#
    return obsd


def _indicators(d: dict, k: str, arr: np.ndarray) -> Dict[str, tuple]:
    """
    handle LLI (loss of lock) and SSI (signal strength)
    """
    if k.startswith(('L1', 'L2')):
        d[k+'lli'] = (('time', 'sv'), np.atleast_2d(arr[:, 0]))

    d[k+'ssi'] = (('time', 'sv'), np.atleast_2d(arr[:, 1]))

    return d


def _gen_array(signalTime: List[datetime], signalConst: List[str], signalPrn: List[str],
               const: List[str], prn: List[str], val: List[Any], valarray: np.array,
               sysname: str) -> xarray.DataArray:
    valarray[:] = np.nan
    for i, (constList, prnList, valList) in enumerate(zip(const, prn, val)):
        constIdx = np.searchsorted(signalConst, constList)
        prnIdx = np.searchsorted(signalPrn, prnList)
        valarray[i, constIdx, prnIdx] = valList

    return xarray.DataArray(valarray, coords=[signalTime, signalConst, signalPrn], dims=['time', 'const', 'prn'], name=sysname)


def obsheader3(f: TextIO):
    """
    Parse Header information and store information in HeaderClass instance
    """
    if isinstance(f, (str, Path)):
        with opener(f, header=True) as h:
            return obsheader3(h)

    # Initialise instance
    hdr = HeaderClass(rinexinfo(f))

    # Read line-by-line
    for ln in f:
        # read RINEX header labels and check if it is relevant
        h = ln[60:80].strip()
        if h == "END OF HEADER":
            break
        if h not in HeaderClass.labelDict.keys():
            continue

        # Find correct method and parse line information
        getattr(hdr, HeaderClass.labelDict[h])(ln[:60])

    return hdr


def filterObs3(hdr: object,
               use: Sequence[str] = None,
               meas: Sequence[str] = None):
    """
    Filter header for constellation and signals selected for use. The HeaderClass instance is not modified.
    Output: selObs - Dictionary with selected constellations and selected signals
            selInd - Dictionary with selected signals
    """

    # Filter constellations
    if use:
        useSet = set(use)
        selObs = {k: hdr.obsType[k] for k in useSet if k in hdr.obsType}
        if not selObs.keys():
            raise KeyError(f'system type {use} not found in RINEX file')
    else:
        selObs = hdr.obsType

    # Get indices of selected signals and filter selObs respectively
    # Delete constellation key if it holds none of the selected signals
    if meas:
        measSet = set(meas)

        selInd = {}
        _selObsKeys = selObs.keys()
        for k in _selObsKeys:
            indList = []
            for m in measSet:
                for i, o in enumerate(selObs[k]):
                    if o.startswith(m):
                        indList.append(i)
            if indList:
                selInd[k] = set(indList)
                obsList = [ selObs[k][i] for i in selInd[k] ]
                selObs[k] = obsList
            else:
                del selObs[k]

        if not selInd.keys():
            raise KeyError(f'measurement type {meas} not found in RINEX file')
    else:
        for k in selObs.keys():
            selInd[k] = list(range(len(selObs[k])))

    return (selObs, selInd)
