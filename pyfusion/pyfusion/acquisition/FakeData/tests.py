"""Test code for data acquisition."""

from pyfusion.test.tests import PfTestBase

# channel names in pyfusion test config file
timeseries_test_channel_1 = "test_timeseries_channel_1"
timeseries_test_channel_2 = "test_timeseries_channel_2"

multichannel_name = "test_multichannel_timeseries"

class CheckFakeDataAcquisition(PfTestBase):
    """Test the fake data acquisition used for testing."""

    def testBaseClasses(self):
        """Make sure FakeDataAcquisition is subclass of Acquisition."""
        from pyfusion.acquisition.FakeData.acq import FakeDataAcquisition
        from pyfusion.acquisition.base import BaseAcquisition
        self.assertTrue(BaseAcquisition in FakeDataAcquisition.__bases__)

    def testGetDataReturnObject(self):
        """Make sure correct data object type is returned"""
        from pyfusion.acquisition.FakeData.acq import FakeDataAcquisition
        from pyfusion import conf

        # make sure the requested data type is returned using config reference
        test_acq = FakeDataAcquisition('test_fakedata')
        from pyfusion.data.timeseries import TimeseriesData
        data_instance_1 = test_acq.getdata(self.shot_number, timeseries_test_channel_1)
        self.assertTrue(isinstance(data_instance_1, TimeseriesData))
        
        # ...and for kwargs
        # read config as dict and pass as kwargs
        config_dict = conf.utils.get_config_as_dict('Diagnostic', timeseries_test_channel_1)
        data_instance_2 = test_acq.getdata(self.shot_number, **config_dict)
        self.assertTrue(isinstance(data_instance_2, TimeseriesData))

        # check that the two signals are the same
        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(data_instance_1.signal,  data_instance_2.signal) 
        assert_array_almost_equal(data_instance_1.timebase,  data_instance_2.timebase) 
        
    def testDeviceConnection(self):
        """Check that using config loads the correct acquisition."""
        from pyfusion.devices.base import Device
        test_device = Device('TestDevice')
        from pyfusion import conf, config
        acq_name = config.pf_get('Device', 'TestDevice', 'acq_name')
        test_acq = conf.utils.import_setting('Acquisition', acq_name, 'acq_class')
        self.assertTrue(isinstance(test_device.acquisition, test_acq))
        # test that device.acq shortcut works
        self.assertEqual(test_device.acquisition, test_device.acq)
        

    def test_get_data(self):
        """Check that we end up with the correct data class starting from Device"""
        from pyfusion import getDevice
        test_device = getDevice(self.listed_device)
        test_data = test_device.acquisition.getdata(self.shot_number, timeseries_test_channel_1)
        from pyfusion.data.timeseries import TimeseriesData
        self.assertTrue(isinstance(test_data, TimeseriesData))


class CheckFakeDataFetchers(PfTestBase):
    """test DataFetcher subclasses for fake data acquisition."""

    def test_base_classes(self):
        from pyfusion.acquisition.base import BaseDataFetcher
        from pyfusion.acquisition.FakeData.fetch import SingleChannelSineFetcher
        self.assertTrue(BaseDataFetcher in SingleChannelSineFetcher.__bases__)

    def test_singlechannelsinedf(self):
        from pyfusion.acquisition.FakeData.fetch import SingleChannelSineFetcher
        n_samples = 1000
        sample_freq=1.e6
        amplitude = 1.0
        frequency = 3.e4
        t0 = 0.0
        test_shot = -1
        output_data_fetcher = SingleChannelSineFetcher(None, test_shot, sample_freq=sample_freq,
                                                  n_samples=n_samples,
                                                  amplitude=amplitude,
                                                  frequency=frequency,
                                                  t0 = t0)
        output_data = output_data_fetcher.fetch()
        from pyfusion.data.timeseries import TimeseriesData
        self.assertTrue(isinstance(output_data, TimeseriesData))
        from numpy import arange, sin, pi
        from numpy.testing import assert_array_almost_equal
        test_timebase = arange(t0, t0+float(n_samples)/sample_freq, 1./sample_freq)
        assert_array_almost_equal(output_data.timebase, test_timebase)
        test_signal = amplitude*sin(2*pi*frequency*test_timebase)
        assert_array_almost_equal(output_data.signal, test_signal)

class CheckMultiChannel(PfTestBase):
    """Would prefer this to be in acquisition/tests.py...., but we are
    using fakedata"""

    def test_list_channels(self):
        test_shot = 12345
        from pyfusion.acquisition.base import MultiChannelFetcher
        fetcher = MultiChannelFetcher(None, test_shot, config_name=multichannel_name)
        self.assertEqual(fetcher.ordered_channel_names(), ['test_timeseries_channel_1', 'test_timeseries_channel_2'])
        
    
    def test_multichannel_single_channels(self):
        from pyfusion.acquisition.FakeData.acq import FakeDataAcquisition
        from pyfusion import config
        test_acq = FakeDataAcquisition('test_fakedata')
        multichannel_data = test_acq.getdata(self.shot_number, multichannel_name)
        channel_1_data = test_acq.getdata(self.shot_number, timeseries_test_channel_1)
        channel_2_data = test_acq.getdata(self.shot_number, timeseries_test_channel_2)
        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(multichannel_data.signal[0,:], channel_1_data.signal)
        assert_array_almost_equal(multichannel_data.signal[1,:], channel_2_data.signal)
        self.assertEqual(channel_1_data.meta.get('shot'), self.shot_number)
        assert_array_almost_equal(multichannel_data.signal.get_channel(0), channel_1_data.signal)

    def test_different_timebase_exception(self):
        pass
    
    def test_multi_multichannel(self):
        pass

    def test_kwargs_passed_to_channels(self):
        pass


