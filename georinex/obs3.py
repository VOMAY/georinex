from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Union, List, Tuple, Any, Sequence
from typing.io import TextIO

import numpy as np
import xarray as xr

from .rio import opener, rinexinfo, HeaderClass
from .common import check_time_interval
"""https://github.com/mvglasow/satstat/wiki/NMEA-IDs"""

try:
    from pymap3d import ecef2geodetic
except ImportError:
    ecef2geodetic = None


def rinexobs3(fn: Union[TextIO, str, Path],
              use: Sequence[str] = None,
              tlim: Tuple[datetime, datetime] = None,
              useindicators: bool = False,
              meas: Sequence[str] = None,
              verbose: bool = False,
              *,
              interval: Union[float, int, timedelta] = None) -> xr.Dataset:
    """
    process RINEX 3 OBS data

    fn: RINEX OBS 3 filename
    use: 'G'  or ['G', 'R'] or similar

    tlim: read between these time bounds
    useindicators: SSI, LLI are output
    meas:  'L1C'  or  ['L1C', 'C1C'] or similar

    interval: allows decimating file read by time e.g. every 5 seconds.
                Useful to speed up reading of very large RINEX files
    """

# %% Check input arguments and initialise
    interval = check_time_interval(interval)

    if isinstance(use, str):
        use = [use]

    if isinstance(meas, str):
        meas = [meas]

    if not use[0].strip():
        use = None

    if not meas[0].strip():
        meas = None

    if tlim is not None and not isinstance(tlim[0], datetime):
        raise TypeError('time bounds are specified as datetime.datetime')

    last_epoch = None

# %% Parsing loop
    with opener(fn) as f:
        try:
            # Read header into HeaderClass instance
            hdr = obsheader3(f)

            # filter signales based on selection via input arguments
            selObs, selInd = filterObs3(hdr, use, meas)

            # Get set of all signals of all constellations
            signalUnion = sorted(set([ signal for constSigList in selObs.values() for signal in constSigList]))

            # Allocate Main internal data buffer
            obsBuf = {}
            for signal in signalUnion:
                if useindicators:
                    obsBuf[signal] = {'time': [], 'const': [], 'prn': [], 'val': [], 'ssi': [], 'lli': []}
                else:
                    obsBuf[signal] = {'time': [], 'const': [], 'prn': [], 'val': []}

        except KeyError:
            return xr.Dataset()

        # %%    Process OBS file
        for ln in f:
            # Check for next epoch
            if not ln.startswith('>'):
                break

            # %%Process Epoch Record line
            try:
                time, in_range = _timeobs(ln, tlim, last_epoch, interval)
            except ValueError:  # garbage between header and RINEX data
                logging.debug(f'garbage detected in {fn}, trying to parse at next time step')
                continue

            # Number of visible satellites this epoch
            nSv = int(ln[33:35])

            # Check if epoch is in selected interval
            if in_range == -1:
                for _ in range(nSv):
                    next(f)
                continue
            if in_range == 1:
                break
            last_epoch = time

            if verbose:
                print(time, end="\r")

            # %% Process observation lines
            obsEpoch = {}
            # Read nSv lines and extract selected data
            for _, epochLine in zip(range(nSv), f):
                # Check if this line starts with an expected constellatin letter
                if epochLine[0] not in hdr.obsType.keys():
                    raise KeyError(f'Unexpected line found in RINEX file')

                obsEpoch = _epoch(obsEpoch, selObs, selInd, epochLine, useindicators)

            # Store selected data of epoch in internal buffer obsBuf
            for signal in obsEpoch:
                obsBuf[signal]['time'].append(time)
                obsBuf[signal]['const'].append(obsEpoch[signal]['const'])
                obsBuf[signal]['prn'].append(obsEpoch[signal]['prn'])
                obsBuf[signal]['val'].append(obsEpoch[signal]['val'])
                if useindicators:
                    obsBuf[signal]['lli'].append(obsEpoch[signal]['lli'])
                    obsBuf[signal]['ssi'].append(obsEpoch[signal]['ssi'])

    # %% Process OBS file Convert internval buffer (dict) to output format (xarray.DataArray)
    # First generate one DataArray per signal, then merge them together
    data = []
    for signal in obsBuf:
        # Get all times of this signal
        signalTime = obsBuf[signal]['time']
        # Get all constalltions with this signal
        signalConst = np.sort(np.array(list(set([const for constEpochList in obsBuf[signal]['const'] for const in constEpochList]))))
        # Get all satellites of this constellations with this signal
        signalPrn = np.sort(np.array(list(set([prn for prnEpochList in obsBuf[signal]['prn'] for prn in prnEpochList]))))
        # Allocate array of ovservations with three dimensions
        signalVal = np.empty((len(signalTime), len(signalConst), len(signalPrn)))

        # Geneate DataArray and append to list
        data.append(_gen_array(signalTime, signalConst, signalPrn,
                               obsBuf[signal]['const'], obsBuf[signal]['prn'], obsBuf[signal]['val'],
                               signalVal, signal))
        if useindicators:
            data.append(_gen_array(signalTime, signalConst, signalPrn,
                                   obsBuf[signal]['const'], obsBuf[signal]['prn'], obsBuf[signal]['lli'],
                                   signalVal, signal+'-lli'))
            data.append(_gen_array(signalTime, signalConst, signalPrn,
                                   obsBuf[signal]['const'], obsBuf[signal]['prn'], obsBuf[signal]['ssi'],
                                   signalVal, signal+'-ssi'))

    # Merge DataArray
    data = xr.merge(data)

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
                         microsecond=int(float(ln[19:29]) % 1 * 1e6))
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

def _epoch(obsEpoch: Dict[str, Any],
           selObs: Dict[str, Any],
           selInd: Dict[str, Any],
           line: str,
           useindicators: bool) -> (Dict[str, Any]):
    """
    processing of each line in epoch (time step)
    """

    # Get Prn and split in constellation and PRN
    const = line[0]
    prn = int(line[1:3])

    # Skip in constellation not selected
    if const not in selObs.keys():
        return obsEpoch

    # parse the selected records into list
    recordStr = line[3:]
    selRecord = [recordStr[i*16:i*16+16] for i in selInd[const]]

    # Store information in buffer dictionary
    for i, signal in enumerate(selObs[const]):
        # Create key if not available. Only consider inidcators if selected
        if signal not in obsEpoch.keys():
            if useindicators:
                obsEpoch[signal] = {'const': [], 'prn': [], 'val': [], 'ssi': [], 'lli': []}
            else:
                obsEpoch[signal] = {'const': [], 'prn': [], 'val': []}

        obsEpoch[signal]['const'].append(const)
        obsEpoch[signal]['prn'].append(prn)
        obsEpoch[signal]['val'].append(float(selRecord[i][:14]) if len(selRecord[i]) >= 14 and selRecord[i].strip() else np.nan)
        if useindicators:

            obsEpoch[signal]['lli'].append(int(selRecord[i][14]) if len(selRecord[i]) >= 15 and selRecord[i][14].strip() else np.nan)
            obsEpoch[signal]['ssi'].append(int(selRecord[i][15]) if len(selRecord[i]) == 16 and selRecord[i][15].strip() else np.nan)

    return obsEpoch


def _gen_array(signalTime: List[datetime], signalConst: List[str], signalPrn: List[str],
               const: List[str], prn: List[str], val: List[Any], valarray: np.array,
               sysname: str) -> xr.DataArray:
    """
    Generate xarray.DataArray from dictionary. Organise data with 3 coordinates:
        - time
        - const(ellation)
        - prn
    """
    # Fill intermediate value numpy array with
    valarray[:] = np.nan

    # organise data into intermediate array.
    # np.searchsorted uses a significant amount of time. Maybe there is a faster alternative.
    for i, (constList, prnList, valList) in enumerate(zip(const, prn, val)):
        constIdx = np.searchsorted(signalConst, constList)
        prnIdx = np.searchsorted(signalPrn, prnList)
        valarray[i, constIdx, prnIdx] = valList

    # Put everything together in xr.DataArray
    return xr.DataArray(valarray, coords=[signalTime, signalConst, signalPrn], dims=['time', 'const', 'prn'], name=sysname)


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
               meas: Sequence[str] = None) -> Tuple[dict, dict]:
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
    selInd = {}
    if meas:
        measSet = set(meas)

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
