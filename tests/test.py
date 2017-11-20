#!/usr/bin/env python
"""
Self-test file, registration case
for OBS RINEX reader
"""
import xarray
from pyrinex import Path
from numpy.testing import assert_allclose,run_module_suite
#
from pyrinex import rinexobs, rinexnav

rdir=Path(__file__).parents[1]
ofn = rdir/'tests/test.nc'
ofn3 = rdir/'tests/test3.nc'

def test_obs():

    truth = xarray.open_dataarray(str(ofn), group='OBS')

    blocks,hdr = rinexobs(rdir/'tests/demo.10o')

    assert_allclose(blocks,truth)


def test_nav2():

    truth = xarray.open_dataarray(str(ofn), group='NAV')
    testnav = rinexnav(rdir/'tests/demo.10n')

    assert_allclose(testnav,truth)


def test_nav3():

    truth = xarray.open_dataarray(str(ofn3), group='NAV')
    testnav = rinexnav(rdir/'tests/demo3.10n')

    assert_allclose(testnav,truth)


if __name__ == '__main__':
    run_module_suite()
