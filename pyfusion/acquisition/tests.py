"""Test code for data acquisition."""

from pyfusion.test.tests import PfTestBase
from pyfusion.data.base import BaseData
from pyfusion.acquisition.base import BaseAcquisition
from pyfusion.conf.utils import get_config_as_dict, import_from_str
from pyfusion.utils.debug import equal_except_for
from pyfusion.acquisition.utils import getAcquisition
from pyfusion.acquisition.FakeData.acq import FakeDataAcquisition
from pyfusion.acquisition.base import BaseDataFetcher

# Add new acquisition modules here for basic module structure test
acquisition_modules = ['FakeData']

from pyfusion.test.tests import PfTestBase

class CheckAcquisitionArgs(PfTestBase):
    """Make sure we get the same result if we use config or kwargs"""

    def testEqualityConfigOrArgs(self):
        """Check that config  and kwarg instantiated Acquisition classes
        are same."""
        acq_from_config = BaseAcquisition('test_baseacq')
        # create a BaseAcquisition instance with keyword args
        config_dict = get_config_as_dict('Acquisition', 'test_baseacq')
        acq_from_kwargs = BaseAcquisition(**config_dict)
        # Acquistion   instantiated  only   from  keywords   won't  have
        # config_name set but should otherwise be equal
        self.assertTrue(equal_except_for(acq_from_config,
                                         acq_from_kwargs, 'config_name'))

    def testAcqAttrsConfig(self):
        """Check that config, kwarg attributes are correctly attached to
        object.
        
        If config is supplied, load config before kwargs.
        """
        config_dict = get_config_as_dict('Acquisition', 'test_baseacq')
        test_acq = BaseAcquisition('test_baseacq')
        for config_arg in config_dict.keys():
            self.assertTrue(hasattr(test_acq, config_arg))
        
    def testAcqAttrsConfigKwargs(self):
        """Check that config, kwarg attributes are correctly attached to
        object.
        
        If config is supplied, load config before kwargs.
        """
        config_dict = get_config_as_dict('Acquisition', 'test_baseacq')
        test_acq = BaseAcquisition('test_baseacq', dummy_var_1 = 5)
        self.assertEqual(test_acq.dummy_var_1, 5)

    def testAcqAttrsKwargs(self):
        """Check that config, kwarg attributes are correctly attached to
        object.
        
        If config is supplied, load config before kwargs.
        """
        test_acq = BaseAcquisition(dummy_var_1 = 5)
        self.assertEqual(test_acq.dummy_var_1, 5)


class CheckGetAcquisition(PfTestBase):
    """test getAcquisition function."""

    def test_get_acquistion(self):
        test_acq_1 = getAcquisition('test_fakedata')
        test_acq_2 = FakeDataAcquisition('test_fakedata')
        self.assertEqual(test_acq_1.__class__, test_acq_2.__class__)
        self.assertEqual(test_acq_1.__dict__, test_acq_2.__dict__)


class CheckAcquisitionModules(PfTestBase):
    """Check for existence of acquisition modules."""

    def check_module(self, module_name):
        path = 'pyfusion.acquisition'
        import_from_str('.'.join([path, module_name]))
        import_from_str('.'.join([path, module_name, 'acq']))
        import_from_str('.'.join([path, module_name, 'fetch']))

    def testModules(self):
        for module_name in acquisition_modules:
            self.check_module(module_name)

class DummyFetcher(BaseDataFetcher):
    def __init__(self, *args, **kwargs):
        self.connected = False
        super(DummyFetcher, self).__init__(*args, **kwargs)
    def setup(self):
        self.connected = True
    def pulldown(self):
        self.connected = False
    def do_fetch(self):
        if self.connected:
            d=BaseData()
            d.meta["connected"]=True
            return d
        else:
            d=BaseData()
            d.meta["connected"]=False
            return d

class CheckDataFetchers(PfTestBase):
    """test DataFetcher subclasses for fake data acquisition."""

    def test_setup_pulldown(self):
        dummy_shot_number = 12345
        test_fetch = BaseDataFetcher(None, dummy_shot_number)
        self.assertTrue(hasattr(test_fetch, 'setup'))
        self.assertTrue(hasattr(test_fetch, 'pulldown'))
        self.assertTrue(hasattr(test_fetch, 'fetch'))
        self.assertTrue(hasattr(test_fetch, 'do_fetch'))
        
    def testDummyFetcher(self):
        dummy_shot_number = 12345
        test_fetcher = DummyFetcher(None, dummy_shot_number)
        self.assertEqual(test_fetcher.connected, False)
        d=test_fetcher.fetch()
        self.assertEqual(d.meta["connected"], True)
        
    def testFetcherShotArg(self):
        dummy_shot_number = 12345
        test_fetcher = DummyFetcher(None, dummy_shot_number)
        self.assertEqual(test_fetcher.shot, dummy_shot_number)


    
