"""Test LHD device code."""

import os
from pyfusion.test.tests import PfTestBase, BasePyfusionTestCase
import pyfusion

TEST_DATA_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_CONFIG_FILE = os.path.join(TEST_DATA_PATH, "test.cfg")

class LHDDeviceTestCase(BasePyfusionTestCase):

    def setUp(self):
        pyfusion.conf.utils.clear_config()
        if pyfusion.orm_manager.IS_ACTIVE:
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()
        pyfusion.conf.utils.read_config(TEST_CONFIG_FILE)


class CheckLHDDevice(LHDDeviceTestCase):
    def test_device(self):
        lhd = pyfusion.getDevice('LHD')
        data = lhd.acq.getdata(90091, 'LHD_Mirnov_toroidal')
CheckLHDDevice.net = True
CheckLHDDevice.lhd = True
