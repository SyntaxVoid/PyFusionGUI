"""Test code for delimiter-separated values (DSV) flat file acquisition."""

import os
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from numpy import genfromtxt, transpose
from pyfusion.acquisition.DSV.acq import DSVAcquisition
from pyfusion.acquisition.base import BaseAcquisition
from pyfusion.test.tests import PfTestBase

class CheckDSVAcquisition(PfTestBase):
    """Test the fake data acquisition used for testing."""

    def testBaseClasses(self):
        """Make sure DSVAcquisition is subclass of Acquisition."""
        self.assertTrue(BaseAcquisition in DSVAcquisition.__bases__)

    def testdata(self):
        """read in a test file and compare returned data object to expected result""" 
        filename = os.path.join(os.path.dirname(__file__), "test_dsv.dat")
        data = genfromtxt(filename, unpack=True)
        test_acq = DSVAcquisition()
        test_shot_number = 12345

        test_data = test_acq.getdata(test_shot_number, filename=filename,
                         data_fetcher='pyfusion.acquisition.DSV.fetch.DSVMultiChannelTimeseriesFetcher')
        assert_array_almost_equal(data[0], test_data.timebase)
        for i, channel_i in enumerate(data[1:]):
            assert_array_almost_equal(channel_i, test_data.signal[i])
        self.assertEqual(test_shot_number, test_data.meta['shot'])

    def test_shot_data(self):
        """Check that filenames with shot numbers are are used when requested."""
        filename = os.path.join(os.path.dirname(__file__), "test_dsv_shot_(shot).dat")
        test_acq = DSVAcquisition()
        test_shot_number = 12345
        test_data = test_acq.getdata(test_shot_number, filename=filename,
                         data_fetcher='pyfusion.acquisition.DSV.fetch.DSVMultiChannelTimeseriesFetcher')

        

CheckDSVAcquisition.dev = True
