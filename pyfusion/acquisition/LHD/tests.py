""" Test code for LHD data acquision."""

from pyfusion.test.tests import PfTestBase

class TestLHDDataAcq(PfTestBase):

    def test_return_type(self):
        from pyfusion.acquisition.LHD.acq import LHDAcquisition
        
TestLHDDataAcq.lhd = True
