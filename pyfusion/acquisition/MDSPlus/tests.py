"""Test code for MDSPlus data acquisition."""
import os

import pyfusion
from pyfusion.test.tests import PfTestBase, BasePyfusionTestCase
from pyfusion.data.base import BaseData
try:
    from pyfusion.acquisition.MDSPlus.fetch import MDSPlusBaseDataFetcher
except:
    # importing  MDSPlusBaseDataFetcher  will  fail  if MDSPlus  is  not
    # installed.   in general,  we  avoid mds  in  nosttests by  calling
    # nosetests -a '!mds'  pyfusion but in this case,  the import occurs
    # outside of a class definition, and can't be avoided with -a '!mds'
    
    # As  a kludge, we'll  put another  class into  the namespace  so we
    # don't get a syntax error when  we subclass it. (We don't care that
    # the subclass isn't what we want  because we're not going to use it
    # anyway if we don't have MDSPlus)
    from pyfusion.data.base import BaseData as MDSPlusBaseDataFetcher


TEST_DATA_PATH = os.path.abspath(os.path.dirname(__file__))
TEST_CONFIG_FILE = os.path.join(TEST_DATA_PATH, "test.cfg")
TEST_MDSPLUS_SERVER = 'localhost:8001'

class DummyMDSData(BaseData):
    pass

class DummyMDSDataFetcher(MDSPlusBaseDataFetcher):
    """Check that we have a mds data object passed though"""
    def do_fetch(self):
        # this breaks unit tests:
        #data = DummyMDSData()
        # this doesn't. Why??
        data = BaseData()
        data.meta['mds_Data'] = self.acq._Data
        return data


class CheckMDSPlusDataAcquisition(PfTestBase):

    def testBaseClasses(self):
        from pyfusion.acquisition.MDSPlus.acq import MDSPlusAcquisition
        from pyfusion.acquisition.base import BaseAcquisition
        self.assertTrue(BaseAcquisition in MDSPlusAcquisition.__bases__)

    def testHaveMDSPlusDataObject(self):
        from pyfusion.acquisition.MDSPlus.acq import MDSPlusAcquisition
        #test_acq = MDSPlusAcquisition(server='h1data.anu.edu.au')
        test_acq = MDSPlusAcquisition(server=TEST_MDSPLUS_SERVER)
        #self.assertTrue(hasattr(test_acq, '_Data'))
        #from MDSplus import Data
        #self.assertEqual(Data.__dict__, test_acq._Data.__dict__)

CheckMDSPlusDataAcquisition.h1 = True
CheckMDSPlusDataAcquisition.mds = True
CheckMDSPlusDataAcquisition.net = True
CheckMDSPlusDataAcquisition.slow = True

class CheckMDSPlusDataFetchers(PfTestBase):

    def testDataFetcherBaseClass(self):
        from pyfusion.acquisition.base import BaseDataFetcher
        from pyfusion.acquisition.MDSPlus.fetch import MDSPlusDataFetcher
        self.assertTrue(BaseDataFetcher in MDSPlusDataFetcher.__bases__)
    
    #def testMDSDataObjectArg(self):
    #    dummy_shot = 12345
    #    from pyfusion.acquisition.MDSPlus.acq import MDSPlusAcquisition
    #    test_acq = MDSPlusAcquisition(server="h1data.anu.edu.au")
    #    test_fetch = DummyMDSDataFetcher(dummy_shot, _Data=test_acq._Data, mds_tree='H1DATA')
    #    test_data_obj = test_fetch.fetch()

    #def testMDSDataObjectAcq(self):
    #    dummy_shot = 12345
    #    from pyfusion.acquisition.MDSPlus.acq import MDSPlusAcquisition
    #    #test_acq = MDSPlusAcquisition(server="h1data.anu.edu.au")
    #    test_acq = MDSPlusAcquisition(server=TEST_MDSPLUS_SERVER)
    #    #test_fetch = DummyMDSDataFetcher(dummy_shot, _Data=None)
    #    #test_data_obj = test_fetch.fetch()
    #    # TODO: should be able to pass either string or module 
    #    df_str = "pyfusion.acquisition.MDSPlus.tests.DummyMDSDataFetcher"
    #    test_data = test_acq.getdata(dummy_shot, data_fetcher=df_str, mds_tree="H1DATA")
    #    from MDSplus import Data
    #    self.assertEqual(Data.__dict__, test_data.meta['mds_Data'].__dict__)
        
CheckMDSPlusDataFetchers.h1 = True
CheckMDSPlusDataFetchers.mds = True
CheckMDSPlusDataFetchers.net = True
CheckMDSPlusDataFetchers.slow = True



class CheckMDSPlusH1Connection(PfTestBase):
    """tests which require access to h1data.anu.edu.au"""

    def testH1TimeseriesData(self):
        from pyfusion.acquisition.MDSPlus.acq import MDSPlusAcquisition
        h1mds = MDSPlusAcquisition(server=TEST_MDSPLUS_SERVER)
        df_str = "pyfusion.acquisition.MDSPlus.fetch.MDSPlusDataFetcher"
        test_data = h1mds.getdata(58133,
                                  data_fetcher = df_str,
                                  mds_path=r"\h1data::top.operations.mirnov:a14_14:input_1")
        from pyfusion.data.timeseries import TimeseriesData
        self.assertTrue(isinstance(test_data, TimeseriesData))
        self.assertEqual(test_data.signal[0], -0.01953125)

CheckMDSPlusH1Connection.h1 = True
CheckMDSPlusH1Connection.mds = True
CheckMDSPlusH1Connection.net = True
CheckMDSPlusH1Connection.slow = True


class MDSAcqTestCase(BasePyfusionTestCase):

    def setUp(self):
        pyfusion.conf.utils.clear_config()
        if pyfusion.orm_manager.IS_ACTIVE:
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()
        pyfusion.conf.utils.read_config(TEST_CONFIG_FILE)


class CheckH1ConfigSection(MDSAcqTestCase):
    """make sure H1 section in config file works"""

     
    def testH1Config(self):
        import pyfusion
        h1 = pyfusion.getDevice('H1')
        test_mirnov = h1.acq.getdata(58133, 'H1_mirnov_array_1_coil_1')
        self.assertEqual(test_mirnov.signal[0], -0.01953125)

        
    def testH1Multichannel(self):
        import pyfusion
        shot = 58133
        diag = "H1_mirnov_array_1"
        #d=pyfusion.getDevice("H1")
        d=pyfusion.devices.base.Device("H1")
        #data=d.acq.getdata(shot, diag)
        
CheckH1ConfigSection.h1 = True
CheckH1ConfigSection.mds = True
CheckH1ConfigSection.net = True
CheckH1ConfigSection.slow = True


class TestRefactoredMDSLocal(MDSAcqTestCase):
    """Test local data access with the refactored MDS interface.

    The refactored MDSplus package uses the same acquisition and fetcher
    classes for  the different MDSplus access modes  (local, thin client
    and thick client).

    The local data access test is in  a separate class so we can flag it
    with  net=False, with  the  other access  modes  being flagged  with
    net=True.
    """

    def test_local_access(self):
        shot = -1
        tree_path = os.path.join(TEST_DATA_PATH, 'test_tree')
        from pyfusion.acquisition.utils import get_acq_from_config
        test_acq_class = get_acq_from_config('test_local_tree')
        test_acq = test_acq_class('test_local_tree', test_tree_path=tree_path)
        #test_acq = pyfusion.getAcquisition('test_tree', 
        test_data = test_acq.getdata(shot, 'test_signal')


class TestRefactoredMDSThin(MDSAcqTestCase):
    """Test thin client data access with the refactored MDS interface.

    The refactored MDSplus package uses the same acquisition and fetcher
    classes for  the different MDSplus access modes  (local, thin client
    and thick client).
    """

    def test_thin_client_access(self):
        shot = -1
        test_acq = pyfusion.getAcquisition('test_thin_client') 
        test_data = test_acq.getdata(shot, 'test_signal')

TestRefactoredMDSThin.net = True

class TestRefactoredMDSThick(MDSAcqTestCase):
    """Test thick client data access with the refactored MDS interface.

    The refactored MDSplus package uses the same acquisition and fetcher
    classes for  the different MDSplus access modes  (local, thin client
    and thick client).
    """

    def test_thick_client_access(self):
        shot = -1
        test_acq = pyfusion.getAcquisition('test_thick_client') 
        test_data = test_acq.getdata(shot, 'test_signal')

TestRefactoredMDSThick.net = True


###
### Tests for web interface
### 

WEBTEST_CONFIG_FILE = os.path.join(TEST_DATA_PATH, "webtest.cfg")

class WebTestCase(BasePyfusionTestCase):
    def setUp(self):
        pyfusion.conf.utils.clear_config()
        if pyfusion.orm_manager.IS_ACTIVE:
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()
        pyfusion.conf.utils.read_config(WEBTEST_CONFIG_FILE)

class TestWebDataAcq(WebTestCase):
    def test_acq(self):
        test_device = pyfusion.getDevice("TestWebDevice")
        test_data = test_device.acq.getdata(58063, "TestMirnovOne")

TestWebDataAcq.net = True
TestWebDataAcq.dev = True
