from datetime import datetime
import gzip
import zipfile
from pathlib import Path
from contextlib import contextmanager
import io
import logging
import xarray
from typing.io import TextIO
import typing

try:
    import unlzw
except ImportError:
    unlzw = None

from .hatanaka import opencrx


@contextmanager
def opener(fn: typing.Union[TextIO, Path],
           header: bool = False) -> TextIO:
    """provides file handle for regular ASCII or gzip files transparently"""
    if isinstance(fn, str):
        fn = Path(fn).expanduser()

    if isinstance(fn, io.StringIO):
        fn.seek(0)
        yield fn
    elif isinstance(fn, Path):
        finf = fn.stat()
        if finf.st_size > 100e6:
            logging.info(f'opening {finf.st_size/1e6} MByte {fn.name}')

        if fn.suffix == '.gz':
            with gzip.open(fn, 'rt') as f:
                version, is_crinex = rinex_version(f.readline(80))
                f.seek(0)

                if is_crinex and not header:
                    f = io.StringIO(opencrx(f))
                yield f
        elif fn.suffix == '.zip':
            with zipfile.ZipFile(fn, 'r') as z:
                flist = z.namelist()
                for rinexfn in flist:
                    with z.open(rinexfn, 'r') as bf:
                        f = io.StringIO(io.TextIOWrapper(bf, encoding='ascii', errors='ignore').read())
                        yield f
        elif fn.suffix == '.Z':
            if unlzw is None:
                raise ImportError('pip install unlzw')
            with fn.open('rb') as zu:
                with io.StringIO(unlzw.unlzw(zu.read()).decode('ascii')) as f:
                    yield f
        else:  # assume not compressed (or Hatanaka)
            with fn.open('r', encoding='ascii', errors='ignore') as f:
                version, is_crinex = rinex_version(f.readline(80))
                f.seek(0)

                if is_crinex and not header:
                    f = io.StringIO(opencrx(f))
                yield f
    else:
        raise OSError(f'Unsure what to do with input of type: {type(fn)}')


def rinexinfo(f: typing.Union[Path, TextIO]) -> typing.Dict[str, typing.Any]:
    """verify RINEX version"""

    if isinstance(f, (str, Path)):
        fn = Path(f).expanduser()

        if fn.suffix == '.nc':
            attrs: typing.Dict[str, typing.Any] = {'rinextype': []}
            for g in ('OBS', 'NAV'):
                try:
                    dat = xarray.open_dataset(fn, group=g)
                    attrs['rinextype'].append(g.lower())
                except OSError:
                    continue
                attrs.update(dat.attrs)
            return attrs

        with opener(fn, header=True) as f:
            return rinexinfo(f)

    f.seek(0)

    try:
        line = f.readline(80)  # don't choke on binary files

        if line.startswith('#c'):
            return {'version': 'c',
                    'rinextype': 'sp3'}

        version = rinex_version(line)[0]
        file_type = line[20]
        if int(version) == 2:
            if file_type == 'N':
                system = 'G'
            elif file_type == 'G':
                system = 'R'
            elif file_type == 'E':
                system = 'E'
            else:
                system = line[40]
        else:
            system = line[40]

        if line[20] in ('O', 'C'):
            rinex_type = 'obs'
        elif line[20] == 'N' or 'NAV' in line[20:40]:
            rinex_type = 'nav'
        else:
            rinex_type = line[20]

        info = {'version': version,
                'filetype': file_type,
                'rinextype': rinex_type,
                'systems': system}

    except (TypeError, AttributeError, ValueError, UnicodeDecodeError) as e:
        # keep ValueError for consistent user error handling
        raise ValueError(f'not a known/valid RINEX file.  {e}')

    return info


def rinex_version(s: str) -> typing.Tuple[typing.Union[float, str], bool]:
    """

    Parameters
    ----------

    s : str
       first line of RINEX/CRINEX file

    Results
    -------

    version : float
        RINEX file version

    is_crinex : bool
        is it a Compressed RINEX CRINEX Hatanaka file
    """
    if not isinstance(s, str):
        raise TypeError('need first line of RINEX file as string')
    if len(s) < 2:
        raise ValueError(f'first line of file is corrupted {s}')

    if len(s) >= 80:
        if s[60:80] not in ('RINEX VERSION / TYPE', 'CRINEX VERS   / TYPE'):
            raise ValueError('The first line of the RINEX file header is corrupted.')

    # %% .sp3 file
    if s[0] == '#':
        if s[1] != 'c':
            raise ValueError('Georinex only handles version C of SP3 files.')
        return 'sp3' + s[1], False
    # %% typical RINEX files
    try:
        vers = float(s[:9])  # %9.2f
    except ValueError as err:
        raise ValueError(f'Could not determine file version from {s[:9]}   {err}')

    is_crinex = s[20:40] == 'COMPACT RINEX FORMAT'

    return vers, is_crinex


class HeaderClass:
    """ Class to interpret and store Rinex Header information
    """

    # Dictionary mapping the RINEX header labels to the correct parsing function
    labelDict = {'RINEX VERSION / TYPE': 'pInfo',
                 'MARKER NAME': 'pMarkerName',
                 'MARKER NUMBER': 'pMarkerNumber',
                 'OBSERVER / AGENCY': 'pObservAgency',
                 'REC # / TYPE / VERS': 'pReceiver',
                 'ANT # / TYPE': 'pAntenna',
                 'APPROX POSITION XYZ': 'pPosition',
                 'SYS / # / OBS TYPES': 'pSignal',
                 'TIME OF FIRST OBS': 'pTimeFirst',
                 'TIME OF LAST OBS': 'pTimeLast',
                 'INTERVAL': 'pInterval'}

    def __init__(self, info):
        """ Init class with info from rinexinfo()
        """
        self.version = info['version']
        self.fileType = info['filetype']
        self.rinexType = info['rinextype']
        self.systems = info['systems']

        self.obsType = dict()

    def pInfo(self, info):
        self.version = info['version']
        self.fileType = info['filetype']
        self.rinexType = info['rinextype']
        self.systems = info['systems']

    def pMarkerName(self, h: str):
        self.markername = h.strip()

    def pMarkerNumber(self, h: str):
        self.markernumber = h.strip()

    def pObservAgency(self, h: str):
        self.observer = h[0:20].strip()
        self.agency = h[20:60].strip()

    def pReceiver(self, h: str):
        self.recNumber = h[0:20].strip()
        self.recType = h[20:40].strip()
        self.recVersion = h[40:60].strip()

    def pAntenna(self, h: str):
        self.antNumber = h[0:20].strip()
        self.antType = h[20:40].strip()

    def pPosition(self, h: str):
        """ Write coordaintes in list and convert to geodetic
        """
        try:
            self.position= [float(x) for x in h.split()]
            try:
                from pymap3d import ecef2geodetic
            except ImportError:
                ecef2geodetic = None
            if ecef2geodetic is not None:
                self.positionGeodetic = ecef2geodetic(*self.position)

        except (KeyError, ValueError):
            self.position = None

    def pTimeFirst(self, h: str):
        try:
            self.tFirst = datetime(year=int(h[:6]), month=int(h[6:12]), day=int(h[12:18]),
                             hour=int(h[18:24]), minute=int(h[24:30]), second=int(float(h[30:36])),
                             microsecond=int(float(h[30:43]) % 1 * 1e6))
        except (KeyError, ValueError):
            self.tFirst = None

    def pTimeLast(self, h: str):
        try:
            self.tLast = datetime(year=int(h[:6]), month=int(h[6:12]), day=int(h[12:18]),
                             hour=int(h[18:24]), minute=int(h[24:30]), second=int(float(h[30:36])),
                             microsecond=int(float(h[30:43]) % 1 * 1e6))
        except (KeyError, ValueError):
            self.tLast = None

    def pInterval(self, h: str):
        try:
            self.interval = float(h[:10])
        except (KeyError, ValueError):
            self.interval = None

    def pSignal(self, h:str):
        """ Get constellations with signals and store in dictionary.
            E.g. {'E': ['C1C', 'L1C', 'S1C', 'C6C'],
                  'R': ['C1C', 'L1C', 'S1C', 'C1P']}
        """
        if h[0].strip():
            self._currConst = h[0]

        if h[3:7].strip():
            self._currNum   = int(h[3:7])
            self.obsType[self._currConst] = []

        self.obsType[self._currConst].extend(h[7:59].split())
