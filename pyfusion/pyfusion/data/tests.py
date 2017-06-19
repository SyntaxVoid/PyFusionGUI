"""Tests for the Data and related classes."""

from numpy.testing import assert_array_almost_equal, assert_almost_equal
import numpy as np

from pyfusion.test.tests import PfTestBase
from pyfusion.data.base import BaseData, DataSet, Coords, ChannelList, Channel, FloatDelta, BaseOrderedDataSet
from pyfusion.data.timeseries import TimeseriesData, Timebase, Signal, FlucStruc, generate_timebase, SVDData
from pyfusion.data.utils import cps, remap_periodic, peak_freq
from pyfusion.data.base import BaseCoordTransform
from pyfusion.data.filters import reduce_time
from pyfusion.orm.utils import orm_register
from pyfusion.acquisition.FakeData.acq import FakeDataAcquisition

from pyfusion.test.tests import PfTestBase
import pyfusion


DEFAULT_N_CHANNELS = 10
DEFAULT_T0 = 0.00
DEFAULT_T1 = 0.01
DEFAULT_SAMPLE = 1.0e-5
DEFAULT_TIMEBASE = Timebase(np.arange(DEFAULT_T0, DEFAULT_T1, DEFAULT_SAMPLE))
DEFAULT_NOISE = 0.2

mode_1 = {'amp': 0.7, 'freq': 24.0e3, 'mode_number':3, 'phase':0.2}
mode_2 = {'amp': 0.5, 'freq': 37.0e3, 'mode_number':4, 'phase':0.3}
mode_3 = {'amp': 0.5, 'freq': 27.0e3, 'mode_number':2, 'phase':0.1}
mode_4 = {'amp': 0.7, 'freq': 24.0e3, 'mode_number':7, 'phase':3.2}
mode_5 = {'amp': 1.0, 'freq': 37.0e3, 'mode_number':9, 'phase':1.3}

###############################################################################
## Convenience functions                                                     ##
###############################################################################

def get_n_channels(n_ch):
    """Return a list of n_ch channels."""
    poloidal_coords = 2*np.pi*np.arange(n_ch)/n_ch
    channel_gen = (Channel('ch_%02d' %i, Coords('cylindrical', (1.0,i,0.0)))
                   for i in poloidal_coords)
    return ChannelList(*channel_gen)


def get_multimode_test_data(channels = get_n_channels(DEFAULT_N_CHANNELS),
                            timebase = DEFAULT_TIMEBASE,
                            modes = [mode_1, mode_2], noise = DEFAULT_NOISE):
    """Generate synthetic multi-channel data for testing."""
    n_channels = len(channels)
    data_size = (n_channels, timebase.size)
    data_array = noise*2*(np.random.random(data_size)-0.5)
    timebase_matrix = np.resize(timebase, data_size)
    angle_matrix = np.resize(np.array([i.coords.cylindrical[1] for i in channels]),
                          data_size[::-1]).T
    for m in modes:
        data_array += m['amp']*np.cos(2*np.pi*m['freq']*timebase_matrix +
                                   m['mode_number']*angle_matrix + m['phase'])
    output = TimeseriesData(timebase=timebase, signal=Signal(data_array),
                            channels=channels)
    return output

###############################################################################
## End of convenience functions                                              ##
###############################################################################

###############################################################################
## Tests for pyfusion/data/utils.py                                          ##
###############################################################################

class CheckUtils(PfTestBase):
    """Test the helper functions in pyfusion.data.utils.py"""
    def test_remap_periodic(self):
        data = np.array([-3,-2,-1,0,1,2,2.5, 3])
        output = remap_periodic(data, min_val = 0, period=3)
        expected = np.array([0, 1, 2, 0, 1, 2, 2.5, 0])
        assert_array_almost_equal(output, expected)

    def test_peak_freq(self):
        timebase = Timebase(np.arange(0.0,0.01, 1.e-6))
        single_mode_signal = get_multimode_test_data(channels=get_n_channels(1),
                                                     timebase=timebase,
                                                     modes = [mode_3])
        p_f, p_f_elmt = peak_freq(single_mode_signal.signal[0],
                                  single_mode_signal.timebase)
        # Check that we get mode_3 frequency of 27.0 kHz (to 1 decimal place).
        self.assertAlmostEqual(1.e-3*p_f, 1.e-3*mode_3['freq'], 1)


###############################################################################
## End of tests for pyfusion/data/utils.py                                   ##
###############################################################################

###############################################################################
## Tests for pyfusion/data/base.py                                           ##
###############################################################################

class DummyCoordTransform(BaseCoordTransform):
    """Minimal coordinate transform class for testing.

    Transforms cylindrical coordinates (a,b,c) to dummy coordinates (2a,3b,4c)

    """
    input_coords = 'cylindrical'
    output_coords = 'dummy'

    def transform(self, coords):
        return (2*coords[0], 3*coords[1], 4*coords[2])


class CheckCoordinates(PfTestBase):
    """Check that we can add and transform coordinates."""    
    def test_add_coords(self):
        dummy_coords = Coords('cylindrical',(1.0,1.0,1.0))
        self.assertEqual(dummy_coords.cylindrical, (1.0,1.0,1.0))
        dummy_coords.add_coords(cartesian=(0.1,0.5,0.2))
        self.assertEqual(dummy_coords.cartesian, (0.1,0.5,0.2))

    def test_coord_transform(self):
        cyl_coords_1 = (1.0,1.0,1.0)
        dummy_coords_1 = Coords('cylindrical',cyl_coords_1)
        dummy_coords_1.load_transform(DummyCoordTransform)
        # The DummyCoordinateTransform should map (a,b,c) -> (2a,3b,4c)
        self.assertEqual(dummy_coords_1.dummy(), (2*cyl_coords_1[0],
                                                  3*cyl_coords_1[1],
                                                  4*cyl_coords_1[2]))
        # Check again with different ccoordinated.
        cyl_coords_2 = (2.0,1.0,4.0)
        dummy_coords_2 = Coords('cylindrical',cyl_coords_2)
        dummy_coords_2.load_transform(DummyCoordTransform)
        self.assertEqual(dummy_coords_2.dummy(), (2*cyl_coords_2[0],
                                                  3*cyl_coords_2[1],
                                                  4*cyl_coords_2[2]))


class CheckChannels(PfTestBase):
    """Make sure that arguments passed to Channel() appear as attributes."""
    def test_channel_class(self):
        test_coords = Coords('cylindrical',(0.0,0.0,0.0))
        test_ch = Channel('test_channel', test_coords)
        self.assertEqual(test_ch.name, 'test_channel')
        self.assertEqual(test_ch.coords, test_coords)


class CheckChannelsSQL(PfTestBase):
    def test_channels_SQL(self):
        test_coords = Coords('cylindrical',(0.0,0.0,0.0))
        test_ch = Channel('test_1', test_coords)
        test_ch.save()
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            our_channel = session.query(Channel).first()
            self.assertEqual(our_channel.name, 'test_1')


class CheckChannelList(PfTestBase):
    def test_channel_list(self):

        ch01 = Channel('test_1', Coords('dummy', (0,0,0)))
        ch02 = Channel('test_2', Coords('dummy', (0,0,0)))
        ch03 = Channel('test_3', Coords('dummy', (0,0,0)))
                       
        new_cl = ChannelList([ch01, ch02, ch03])

    def test_channellist_ORM(self):

        ch01 = Channel('test_1', Coords('dummy', (0,0,0)))
        ch02 = Channel('test_2', Coords('dummy', (0,0,0)))
        ch03 = Channel('test_3', Coords('dummy', (0,0,0)))

        new_cl = ChannelList(ch03, ch01, ch02)

        new_cl.save()

        # get our channellist
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            our_channellist = session.query(ChannelList).order_by("id").first()

            self.assertEqual(our_channellist[0].name, 'test_3')
            self.assertEqual(our_channellist[1].name, 'test_1')
            self.assertEqual(our_channellist[2].name, 'test_2')


class CheckDataSet(PfTestBase):

    def test_dataset(self):
        ch=get_n_channels(5)
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd_1 = TimeseriesData(timebase=tb, signal=Signal(np.resize(np.arange(5*len(tb)),(5,len(tb)))),
                               channels=ch)
        tsd_2 = TimeseriesData(timebase=tb,
                               signal=Signal(np.resize(np.arange(5*len(tb))+1, (5,len(tb)))),
                               channels=ch)
        test_dataset = DataSet('test_ds_1')
        test_dataset.add(tsd_1)
        test_dataset.add(tsd_2)
        self.assertTrue(tsd_1 in test_dataset)
        """
        # we don't support removing items from dataset yet...
        test_dataset.remove(tsd_1)
        self.assertFalse(tsd_1 in test_dataset)
        self.assertTrue(tsd_2 in test_dataset)
        """

    def test_dataset_filters_2(self):
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        ch = get_n_channels(5)
        tsd_1 = TimeseriesData(timebase=tb, signal=Signal(np.resize(np.arange(5*len(tb)),(5,len(tb)))),
                               channels=ch)
        tsd_2 = TimeseriesData(timebase=tb,
                               signal=Signal(np.resize(np.arange(5*len(tb))+1,(5,len(tb)))),
                               channels=ch)
        test_dataset = DataSet('test_ds_2')
        test_dataset.add(tsd_1)
        test_dataset.add(tsd_2)
        test_dataset.reduce_time(new_times)


class CheckOrderedDataSet(PfTestBase):
    """test the ordered dataset"""

    ## need to fix for datasetitems..
    """
    def test_ordereddataset(self):
        #pretend these are datapoints
        class CheckData(BaseData):
            def __init__(self, a, b):
                self.a = a
                self.b = b
                super(TestData, self).__init__()

        d1=TestData(1,2)
        d2=TestData(2,1)

        ods = BaseOrderedDataSet('test_ods')
        ods.append(d1)
        ods.append(d2)

        self.assertEqual(ods[0], d1)
        self.assertEqual(ods[1], d2)

    """ 

    """
    def test_submethod(self):
        class CheckData(BaseData):
            def __init__(self, a):
                self.a = a
                super(TestData, self).__init__()

        d1 = TestData(TestData(1))
        d2 = TestData(TestData(2))
        d3 = TestData(TestData(3))

        ds = OrderedDataSet(ordered_by='a.a')
        for d in [d3, d1, d2]:
            ds.add(d)
        self.assertEqual(ds[0].a.a, 1)
        self.assertEqual(ds[1].a.a, 2)
        self.assertEqual(ds[2].a.a, 3)
    """
    def test_ordered_dataset_ORM(self):

        channel_01 = Channel('channel_01', Coords('dummy', (0,0,0)))
        channel_02 = Channel('channel_02', Coords('dummy', (0,0,0)))
        channel_03 = Channel('channel_03', Coords('dummy', (0,0,0)))
        channel_04 = Channel('channel_04', Coords('dummy', (0,0,0)))
        

        fd1 = FloatDelta(channel_01, channel_02, 0.45)
        fd2 = FloatDelta(channel_02, channel_03, 0.25)
        fd3 = FloatDelta(channel_03, channel_04, 0.49)

        #ods = OrderedDataSet(ordered_by="channel_1.name")
        ods = BaseOrderedDataSet('test_ods')
        
        for fd in [fd3, fd1, fd2]:
            ods.append(fd)

        ods.save()

        # now read out of database
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            db_ods = session.query(BaseOrderedDataSet).first()
            self.assertEqual(db_ods[0].channel_1.name, 'channel_03')
            self.assertEqual(db_ods[1].channel_1.name, 'channel_01')
            self.assertEqual(db_ods[2].channel_1.name, 'channel_02')


class CheckFloatDelta(PfTestBase):
    """delta phase data class."""

    def test_d_phase(self):
        channel_01 = Channel('channel_01', Coords('dummy', (0,0,0)))
        channel_02 = Channel('channel_02', Coords('dummy', (0,0,0)))

        fd = FloatDelta(channel_01, channel_02, 0.45)
        
    def test_ORM_floatdelta(self):
        """ check that floatdelta can be saved to database"""
        channel_01 = Channel('channel_01', Coords('dummy', (0,0,0)))
        channel_02 = Channel('channel_02', Coords('dummy', (0,0,0)))
        fd = FloatDelta(channel_01, channel_02, 0.45)
        fd.save()
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            db_fd = session.query(FloatDelta).first()
            self.assertEqual(db_fd.delta, 0.45)

###############################################################################
## End of tests for pyfusion/data/base.py                                    ##
###############################################################################

###############################################################################
## Tests for pyfusion/data/timeseries.py                                     ##
###############################################################################






###############################################################################
## End of tests for pyfusion/data/timeseries.py                              ##
###############################################################################

##### unsorted tests

class CheckTimeseriesData(PfTestBase):
    """Test timeseries data"""
    def testBaseClasses(self):
        self.assertTrue(BaseData in TimeseriesData.__bases__)


    def test_timebase_and_coords(self):
        n_ch = 10
        n_samples = 1024
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)


class CheckTimebase(PfTestBase):
    """Test Timebase class."""


    def test_timebase(self):
        t0=0.3
        n_samples=500
        sample_freq=1.e6
        test_tb = generate_timebase(t0=t0,n_samples=n_samples, sample_freq=sample_freq)
        local_tb = np.arange(t0, t0+n_samples/sample_freq, 1./sample_freq)
        self.assertTrue((test_tb == local_tb).all())
        self.assertAlmostEqual(test_tb.sample_freq, sample_freq, 4)

    def test_timebase_slice(self):
        t0=0.3
        n_samples=500
        sample_freq=1.e6
        test_tb = generate_timebase(t0=t0,n_samples=n_samples, sample_freq=sample_freq)

        self.assertTrue(hasattr(test_tb, 'sample_freq'))

        sliced_tb = test_tb[:10]

        self.assertTrue(hasattr(sliced_tb, 'sample_freq'))


class CheckSignal(PfTestBase):
    """Test Signal class."""
    
    def test_base_class(self):
        self.assertTrue(np.ndarray in Signal.__bases__)

    def test_n_channels(self):
        test_sig_1a = Signal(np.random.rand(10))
        self.assertEqual(test_sig_1a.n_channels(), 1)
        test_sig_1b = Signal(np.random.rand(1,10))
        self.assertEqual(test_sig_1b.n_channels(), 1)
        test_sig_2 = Signal(np.random.rand(2,10))
        self.assertEqual(test_sig_2.n_channels(), 2)
        
    def test_n_samples(self):
        test_sig_1a = Signal(np.random.rand(10))
        self.assertEqual(test_sig_1a.n_samples(), 10)
        test_sig_1b = Signal(np.random.rand(1,10))
        self.assertEqual(test_sig_1b.n_samples(), 10)
        test_sig_2 = Signal(np.random.rand(2,10))
        self.assertEqual(test_sig_2.n_samples(), 10)

class CheckFilters(PfTestBase):

    def test_reduce_time_filter_single_channel(self):
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.arange(len(tb))),
                             channels=get_n_channels(1))
        new_time_args = np.searchsorted(tb, new_times)
        timebase_test = tsd.timebase[new_time_args[0]:new_time_args[1]].copy()
        signal_test = tsd.signal[new_time_args[0]:new_time_args[1]].copy()
        reduced_tsd = reduce_time(tsd, new_times)
        self.assertTrue(isinstance(reduced_tsd, TimeseriesData))
        assert_array_almost_equal(reduced_tsd.timebase, timebase_test)
        assert_array_almost_equal(reduced_tsd.signal, signal_test)
        
    def test_reduce_time_filter_multi_channel(self):
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.resize(np.arange(5*len(tb)),(5,len(tb)))),
                             channels=get_n_channels(5))
        new_time_args = np.searchsorted(tb, new_times)
        timebase_test = tsd.timebase[new_time_args[0]:new_time_args[1]].copy()
        signal_test = tsd.signal[:,new_time_args[0]:new_time_args[1]].copy()
        reduced_tsd = reduce_time(tsd, new_times)
        self.assertTrue(isinstance(reduced_tsd, TimeseriesData))
        assert_array_almost_equal(reduced_tsd.timebase, timebase_test)
        assert_array_almost_equal(reduced_tsd.signal, signal_test)
    
    def test_reduce_time_filter_multi_channel_attached_method(self):
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.resize(np.arange(5*len(tb)),(5,len(tb)))),
                             channels=get_n_channels(5))
        new_time_args = np.searchsorted(tb, new_times)
        timebase_test = tsd.timebase[new_time_args[0]:new_time_args[1]].copy()
        signal_test = tsd.signal[:,new_time_args[0]:new_time_args[1]].copy()
        reduced_tsd = tsd.reduce_time(new_times)
        self.assertTrue(isinstance(reduced_tsd, TimeseriesData))
        assert_array_almost_equal(reduced_tsd.timebase, timebase_test)
        assert_array_almost_equal(reduced_tsd.signal, signal_test)
    

    def test_reduce_time_dataset(self):
        new_times = [-0.25, 0.25]
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd_1 = TimeseriesData(timebase=tb,
                               signal=Signal(np.resize(np.arange(5*len(tb)),(5,len(tb)))),
                               channels=get_n_channels(5))
        tsd_2 = TimeseriesData(timebase=tb, signal=Signal(np.resize(np.arange(5*len(tb))+1,(5,len(tb)))),
                               channels=get_n_channels(5))
        test_dataset = DataSet('test_dataset')
        test_dataset.add(tsd_1)
        test_dataset.add(tsd_2)
        test_dataset.reduce_time(new_times)
        

class CheckSegmentFilter(PfTestBase):
    
    def test_single_channel_timeseries(self):
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.arange(len(tb))), channels=get_n_channels(1))
        seg_dataset = tsd.segment(n_samples=10)
        self.assertTrue(len(seg_dataset)==10)

    def test_multi_channel_timeseries(self):
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.resize(np.arange(3*len(tb)), (3,len(tb)))),
                             channels=get_n_channels(3))
        seg_dataset = tsd.segment(n_samples=10)
        self.assertTrue(len(seg_dataset)==10)

    def test_dataset(self):
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tsd_1 = TimeseriesData(timebase=tb,
                               signal=Signal(np.resize(np.arange(3*len(tb)), (3,len(tb)))),
                               channels=get_n_channels(3))
        tsd_2 = TimeseriesData(timebase=tb,
                               signal=Signal(np.resize(np.arange(3*len(tb)+1),(3,len(tb)))),
                               channels=get_n_channels(3))
        input_dataset = DataSet('test_dataset')
        input_dataset.add(tsd_1)
        input_dataset.add(tsd_2)
        seg_dataset = input_dataset.segment(n_samples=10)
        self.assertTrue(len(seg_dataset)==20)



"""
class CheckFlucstrucs(PfTestBase):

    def test_fakedata_single_shot(self):
                #d=pyfusion.getDevice('H1')
        #data = d.acq.getdata(58073, 'H1_mirnov_array_1')
        #fs_data = data.reduce_time([0.030,0.031])
"""     

class CheckNormalise(PfTestBase):

    def test_single_channel_fakedata(self):
        test_acq = FakeDataAcquisition('test_fakedata')
        channel_data = test_acq.getdata(self.shot_number, "test_timeseries_channel_2")
        channel_data_norm_no_arg = test_acq.getdata(self.shot_number, "test_timeseries_channel_2").normalise()
        channel_data_rms_norm_by_arg = test_acq.getdata(self.shot_number, "test_timeseries_channel_2").normalise(method='rms')
        channel_data_peak_norm_by_arg = test_acq.getdata(self.shot_number, "test_timeseries_channel_2").normalise(method='peak')
        channel_data_var_norm_by_arg = test_acq.getdata(self.shot_number, "test_timeseries_channel_2").normalise(method='var')
        rms_value = np.sqrt(np.mean(channel_data.signal**2))
        peak_value = max(abs(channel_data.signal))
        var_value = np.var(channel_data.signal)


        assert_array_almost_equal(channel_data.signal/rms_value, channel_data_rms_norm_by_arg.signal)
        assert_array_almost_equal(channel_data.signal/peak_value, channel_data_peak_norm_by_arg.signal)
        assert_array_almost_equal(channel_data.signal/var_value, channel_data_var_norm_by_arg.signal)

        # check that default is peak
        assert_array_almost_equal(channel_data_peak_norm_by_arg.signal, channel_data_norm_no_arg.signal)


        # try for dataset
        channel_data_for_set = test_acq.getdata(self.shot_number, "test_timeseries_channel_2")

        test_dataset = DataSet('test_dataset')
        test_dataset.add(channel_data_for_set)
        test_dataset.normalise(method='rms')
        for d in test_dataset:
            assert_array_almost_equal(channel_data.signal/rms_value, d.signal)
            

        
    def test_multichannel_fakedata(self):
        test_acq = FakeDataAcquisition('test_fakedata')
        multichannel_data = test_acq.getdata(self.shot_number, "test_multichannel_timeseries")

        mcd_ch_0 = multichannel_data.signal.get_channel(0)
        mcd_ch_0_peak = max(abs(mcd_ch_0))
        mcd_ch_0_rms = np.sqrt(np.mean(mcd_ch_0**2))
        mcd_ch_0_var = np.var(mcd_ch_0)
        
        mcd_ch_1 = multichannel_data.signal.get_channel(1)
        mcd_ch_1_peak = max(abs(mcd_ch_1))
        mcd_ch_1_rms = np.sqrt(np.mean(mcd_ch_1**2))
        mcd_ch_1_var = np.var(mcd_ch_1)
        
        mcd_peak_separate = test_acq.getdata(self.shot_number,
                                             "test_multichannel_timeseries").normalise(method='peak', separate=True)
        mcd_peak_whole = test_acq.getdata(self.shot_number,
                                          "test_multichannel_timeseries").normalise(method='peak', separate=False)
        mcd_rms_separate = test_acq.getdata(self.shot_number,
                                            "test_multichannel_timeseries").normalise(method='rms', separate=True)
        mcd_rms_whole = test_acq.getdata(self.shot_number,
                                         "test_multichannel_timeseries").normalise(method='rms', separate=False)
        mcd_var_separate = test_acq.getdata(self.shot_number,
                                            "test_multichannel_timeseries").normalise(method='var', separate=True)
        mcd_var_whole = test_acq.getdata(self.shot_number,
                                         "test_multichannel_timeseries").normalise(method='var', separate=False)
        
        # peak - separate
        assert_array_almost_equal(mcd_peak_separate.signal.get_channel(0), mcd_ch_0/mcd_ch_0_peak)
        assert_array_almost_equal(mcd_peak_separate.signal.get_channel(1), mcd_ch_1/mcd_ch_1_peak)
        # peak - whole
        max_peak = max(mcd_ch_0_peak, mcd_ch_1_peak)
        assert_array_almost_equal(mcd_peak_whole.signal.get_channel(0), mcd_ch_0/max_peak)
        assert_array_almost_equal(mcd_peak_whole.signal.get_channel(1), mcd_ch_1/max_peak)
        
        # rms - separate
        assert_array_almost_equal(mcd_rms_separate.signal.get_channel(0), mcd_ch_0/mcd_ch_0_rms)
        assert_array_almost_equal(mcd_rms_separate.signal.get_channel(1), mcd_ch_1/mcd_ch_1_rms)
        # rms - whole
        max_rms = max(mcd_ch_0_rms, mcd_ch_1_rms)
        assert_array_almost_equal(mcd_rms_whole.signal.get_channel(0), mcd_ch_0/max_rms)
        assert_array_almost_equal(mcd_rms_whole.signal.get_channel(1), mcd_ch_1/max_rms)

        # var - separate
        assert_array_almost_equal(mcd_var_separate.signal.get_channel(0), mcd_ch_0/mcd_ch_0_var)
        assert_array_almost_equal(mcd_var_separate.signal.get_channel(1), mcd_ch_1/mcd_ch_1_var)
        # var - whole
        max_var = max(mcd_ch_0_var, mcd_ch_1_var)
        assert_array_almost_equal(mcd_var_whole.signal.get_channel(0), mcd_ch_0/max_var)
        assert_array_almost_equal(mcd_var_whole.signal.get_channel(1), mcd_ch_1/max_var)



class CheckFlucstrucs(PfTestBase):

    def test_svd_data(self):
        n_ch = 10
        n_samples = 1024
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%02d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)

        test_svd = multichannel_data.svd()
        self.assertTrue(isinstance(test_svd, SVDData))
        self.assertEqual(len(test_svd.topos[0]), n_ch)
        self.assertEqual(len(test_svd.chronos[0]), n_samples)
        assert_array_almost_equal(test_svd.chrono_labels, timebase)
        for c_i, ch in enumerate(channels):
            self.assertEqual(ch, test_svd.channels[c_i])

    def test_SVDData_class(self):
        n_ch = 5
        n_samples = 512
        fake_data = np.resize(np.arange(n_ch*n_samples), (n_ch, n_samples))
        numpy_svd = np.linalg.svd(fake_data, 0)
        test_svd  = SVDData(np.arange(n_samples), np.arange(n_ch), numpy_svd)
        assert_array_almost_equal(test_svd.topos, np.transpose(numpy_svd[0]))
        assert_array_almost_equal(test_svd.svs, numpy_svd[1])
        assert_array_almost_equal(test_svd.chronos, numpy_svd[2])

        E = sum(numpy_svd[1]*numpy_svd[1])
        self.assertEqual(test_svd.E, E)
        p = np.array([i**2 for i in test_svd.svs])/test_svd.E
        assert_array_almost_equal(test_svd.p, p)
        
        self_cps = test_svd.self_cps()
        assert_array_almost_equal(self_cps, np.array([(0.99999999999999989+0j),
                                                   (0.99999999999999922+0j),
                                                   (0.99999999999999767+0j),
                                                   (0.99999999999999867+0j),
                                                   (1.0000000000000002+0j)])
                                  )

    def test_flucstruc_signals(self):
        # make sure that flucstruc derived from all singular values
        # gives back the original signal
        n_ch = 10
        n_samples = 1024
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.01)
        svd_data = multichannel_data.svd()
        test_fs = FlucStruc(svd_data, range(len(svd_data.svs)), multichannel_data.timebase)

        assert_almost_equal(test_fs.signal, multichannel_data.signal)

    def test_flucstruc_phases(self):
        n_ch = 10
        n_samples = 1024
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.01)
        fs_data = multichannel_data.flucstruc(min_dphase = -2*np.pi)
        self.assertTrue(isinstance(fs_data, DataSet))
        self.assertTrue(len([i for i in fs_data.data]) > 0)
        E = 0.7**2 + 0.5**2
        for fs in fs_data:
            self.assertTrue(isinstance(fs, FlucStruc))
            # fs_data is not ordered, so we identify flucstrucs by the sv indicies
            if fs.svs == [0,1]:
                # check that freq is correct to within 1kHz
                self.assertAlmostEqual(1.e-4*fs.freq, 1.e-4*24.e3, 1)
                # 
                fake_phases = -3.0*2*np.pi*np.arange(n_ch+1)[:-1]/(n_ch)
                fake_dphases = fake_phases[1:]-fake_phases[:-1]

                test_dphase = fs.dphase
                # check phases within 0.5 rad
                assert_array_almost_equal(test_dphase, fake_dphases, 1)
                # check fs energy is correct to 3 decimal places
                self.assertAlmostEqual(fs.p, 0.7**2/E, 3)
            if fs.svs == [2,3]:
                self.assertAlmostEqual(1.e-4*fs.freq, 1.e-4*37.e3, 1)
                fake_phases = -4.0*2*np.pi*np.arange(n_ch+1)[:-1]/(n_ch)
                fake_dphases = fake_phases[1:]-fake_phases[:-1]

                test_dphase = fs.dphase
                # check phases within 0.5 rad
                assert_array_almost_equal(test_dphase, fake_dphases, 1)
                # check fs energy is correct to 3 decimal places
                self.assertAlmostEqual(fs.p, 0.5**2/E, 3)
        

    def test_ORM_flucstrucs(self):
        """ check that flucstrucs can be saved to database"""
        n_ch = 10
        n_samples = 1024
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.01)
        # produce a dataset of flucstrucs
        #print ">> ", multichannel_data.channels
        fs_data = multichannel_data.flucstruc(min_dphase = -2*np.pi)
        print type(fs_data)
        #print list(fs_data)[0].dphase[0].channel_1
        #print '---'
        # save our dataset to the database
        fs_data.save()
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            d1 = DataSet('test_dataset_1')
            d1.save()
            d2 = DataSet('test_dataset_2')
            d2.save()

            # get our dataset from database
            our_dataset = session.query(DataSet).order_by("id").first()
            self.assertEqual(our_dataset.created, fs_data.created)

            self.assertEqual(len([i for i in our_dataset.data]), len(our_dataset))

            #check flucstrucs have freq, t0 and d_phase..
            #for i in our_dataset.data:
            #    print i
            #print 'w'
            #assert False

            #our guinea pig flucstruc:
            test_fs = our_dataset.pop()
            self.assertTrue(isinstance(test_fs.freq, float))
            self.assertTrue(isinstance(test_fs.t0, float))

            # now, are the phase data correct?

            self.assertTrue(isinstance(test_fs.dphase, BaseOrderedDataSet))
            self.assertEqual(len(test_fs.dphase), n_ch-1)

            # what if we close the session and try again?

            session.close()
            session = pyfusion.orm_manager.Session()

            ds_again = session.query(DataSet).order_by("id").first()
            fs_again = list(ds_again)[0]
            """
            for i in fs_again.dphase:
                print i
            assert False
            """



        
class CheckRemoveNonContiguousFilter(PfTestBase):

    def test_remove_noncontiguous(self):
        tb1 = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tb2 = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        tb3 = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        # nonzero signal mean
        tsd1 = TimeseriesData(timebase=tb1,
                              signal=Signal(np.arange(len(tb1))), channels=ChannelList(Channel('ch_01',Coords('dummy',(0,0,0)))))
        tsd2 = TimeseriesData(timebase=tb2,
                              signal=Signal(np.arange(len(tb2))), channels=ChannelList(Channel('ch_01',Coords('dummy',(0,0,0)))))
        tsd3 = TimeseriesData(timebase=tb3,
                              signal=Signal(np.arange(len(tb3))), channels=ChannelList(Channel('ch_01',Coords('dummy',(0,0,0)))))

        self.assertTrue(tb1.is_contiguous())
        self.assertTrue(tb2.is_contiguous())
        self.assertTrue(tb3.is_contiguous())
        tsd2.timebase[-50:] += 1.0
        self.assertFalse(tb2.is_contiguous())

        ds = DataSet('ds')
        for tsd in [tsd1, tsd2, tsd3]:
            ds.add(tsd)
        
        for tsd in [tsd1, tsd2, tsd3]:
            self.assertTrue(tsd in ds)

        filtered_ds = ds.remove_noncontiguous()
        for tsd in [tsd1, tsd3]:
            self.assertTrue(tsd in filtered_ds)
            
        self.assertFalse(tsd2 in filtered_ds)


        
class CheckSubtractMeanFilter(PfTestBase):
    """Test mean subtraction filter for timeseries data."""


    def test_remove_mean_single_channel(self):
        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        # nonzero signal mean
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.arange(len(tb))), channels=ChannelList(Channel('ch_01',Coords('dummy',(0,0,0)))))

        filtered_tsd = tsd.subtract_mean()

        assert_almost_equal(np.mean(filtered_tsd.signal), 0)
        
    
    def test_remove_mean_multichanel(self):
        multichannel_data = get_multimode_test_data(channels=get_n_channels(10),
                                                    timebase = Timebase(np.arange(0.0,0.01,1.e-5)),
                                                    noise = 0.2)
        # add some non-zero offset
        multichannel_data.signal += np.random.rand(*multichannel_data.signal.shape)

        filtered_data = multichannel_data.subtract_mean()
        mean_filtered_data = np.mean(filtered_data.signal, axis=1)
        assert_array_almost_equal(mean_filtered_data, np.zeros_like(mean_filtered_data))


    def test_remove_mean_dataset(self):
        multichannel_data_1 = get_multimode_test_data(channels=get_n_channels(10),
                                                      timebase = Timebase(np.arange(0.0,0.01,1.e-5)),
                                                      modes = [mode_1, mode_2],
                                                      noise = 0.2)
        multichannel_data_2 = get_multimode_test_data(channels=get_n_channels(15),
                                                      timebase = Timebase(np.arange(0.0,0.01,1.e-5)),
                                                      modes = [mode_4, mode_5],
                                                      noise = 0.7)
        multichannel_data_3 = get_multimode_test_data(channels=get_n_channels(13),
                                                      timebase = Timebase(np.arange(0.0,0.01,1.e-5)),
                                                      modes = [mode_4, mode_5],
                                                      noise = 0.7)
        # add some non-zero offset
        multichannel_data_1.signal += np.random.rand(*multichannel_data_1.signal.shape)
        multichannel_data_2.signal += np.random.rand(*multichannel_data_2.signal.shape)

        test_dataset = pyfusion.data.base.DataSet('test_dataset')

        test_dataset.add(multichannel_data_1)
        test_dataset.add(multichannel_data_2)


        filtered_data = test_dataset.subtract_mean()
        for d in filtered_data:
            mean_filtered_data = np.mean(d.signal, axis=1)
            assert_array_almost_equal(mean_filtered_data, np.zeros_like(mean_filtered_data))


class CheckFilterMetaClass(PfTestBase):

    def test_new_filter(self):

        # add some filters
        @pyfusion.data.filters.register("CheckData")
        def test_filter(self):
            return self

        @pyfusion.data.filters.register("CheckData", "CheckData2")
        def other_test_filter(self):
            return self

        # now create TestData 

        class CheckData(pyfusion.data.base.BaseData):
            pass

        if pyfusion.orm_manager.IS_ACTIVE:

            @orm_register()
            def orm_load_floatdelta(man):
                from sqlalchemy import Table, Column, Integer, ForeignKey
                from sqlalchemy.orm import mapper
                man.checkdata_table = Table('checkdata', man.metadata,
                                            Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
                # man.metadata.create_all()
                mapper(CheckData, man.checkdata_table, inherits=BaseData, polymorphic_identity='checkdata')
            
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()
            pyfusion.orm_manager.load_orm()
        test_data = CheckData()
        for attr_name in ["test_filter", "other_test_filter"]:
            self.assertTrue(hasattr(test_data, attr_name))

class CheckNumpyFilters(PfTestBase):

    def test_correlate(self):
        multichannel_data = get_multimode_test_data(channels=get_n_channels(2),
                                                    timebase = Timebase(np.arange(0.0,0.01,1.e-5)),
                                                    noise = 0.2)
        numpy_corr = np.correlate(multichannel_data.signal[0], multichannel_data.signal[1])

        pyfusion_corr = multichannel_data.correlate(0,1)
        assert_array_almost_equal(numpy_corr, pyfusion_corr)


class CheckPlotMethods(PfTestBase):
    def test_svd_plot(self):
        n_ch = 4
        n_samples = 256
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.5)

        test_svd = multichannel_data.svd()
        self.assertTrue(hasattr(test_svd, 'svdplot'))
        

class CheckDataHistory(PfTestBase):
    def testNewData(self):
        test_data = BaseData()
        self.assertEqual(test_data.history.split('> ')[1], 'New BaseData')

    def testFilteredDataHistory_nocopy(self):

        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        # nonzero signal mean
        ch = get_n_channels(1)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.arange(len(tb))), channels=ch)

        filtered_tsd = tsd.subtract_mean()
        self.assertEqual(len(filtered_tsd.history.split('\n')), 3)
        output_data = filtered_tsd.normalise(method='rms', copy=False)
        self.assertEqual(filtered_tsd.history.split('> ')[-1], "normalise(method='rms')")
        self.assertEqual(output_data.history.split('> ')[-1], "normalise(method='rms')")

    def testFilteredDataHistory_copy(self):

        tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
        # nonzero signal mean
        ch = get_n_channels(1)
        tsd = TimeseriesData(timebase=tb,
                             signal=Signal(np.arange(len(tb))), channels=ch)

        filtered_tsd = tsd.subtract_mean()
        self.assertEqual(len(filtered_tsd.history.split('\n')), 3)
        output_data = filtered_tsd.normalise(method='rms', copy=True)
        self.assertEqual(output_data.history.split('> ')[-1], "normalise(method='rms')")
        self.assertEqual(filtered_tsd.history.split('> ')[-1], "subtract_mean()")


class CheckDataSetLabels(PfTestBase):
    def test_dataset_label(self):
        test_ds = DataSet('test_ds_1')
        test_ds.save()
        self.assertEqual(test_ds.label, 'test_ds_1')
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            db_ods = session.query(DataSet).filter_by(label='test_ds_1')
            
    def test_baseordereddataset_label(self):
        test_ds = BaseOrderedDataSet('test_ods_1')
        test_ds.save()
        self.assertEqual(test_ds.label, 'test_ods_1')
        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            db_ods = session.query(BaseOrderedDataSet).filter_by(label='test_ods_1')
            
        
        
class CheckGetCoords(PfTestBase):
    def test_get_coords_for_channel_config(self):
        channel_name = "Test_H1_diag"
        coords = pyfusion.data.base.get_coords_for_channel(channel_name)
        self.assertTrue(isinstance(coords, Coords))
        self.assertEqual(coords.default_name, 'cylindrical')


class CheckStoredMetaDataForDataSets(PfTestBase):
    def test_stored_metadata_datasets(self):
        """Make sure metadata attached to dataset classes is saved to sql."""
        n_ch = 3
        n_samples = 1024
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.01)
        # put in some fake metadata 
        multichannel_data.meta = {'hello':'world'}
        #print multichannel_data.meta


        # produce a dataset of flucstrucs
        fs_data = multichannel_data.flucstruc(min_dphase = -2*np.pi)

        # check that metadata is carried to the flucstrucs

        self.assertEqual(fs_data.meta, multichannel_data.meta)

        # save our dataset to the database
        fs_data.save()

        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            some_ds = session.query(DataSet).all().pop()
            self.assertEqual(some_ds.meta, multichannel_data.meta)

        #print some_ds.meta
        #assert False
    
class CheckStoredMetaData(PfTestBase):
    def test_stored_metadata_data(self):
        """ metadata should be stored to data instances, rather than datasets - this might be slower, but more likely to guarantee data is kept track of."""
        n_ch = 3
        n_samples = 1024
        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*1.e-6),
                                                    noise = 0.01)




        # put in some fake metadata 
        multichannel_data.meta = {'hello':'world'}
        print multichannel_data.meta

        # produce a dataset of flucstrucs
        fs_data = multichannel_data.flucstruc(min_dphase = -2*np.pi)

        # check that metadata is carried to the individual flucstrucs
        for fs in fs_data:
            self.assertEqual(fs.meta, multichannel_data.meta)

        # save our dataset to the database
        fs_data.save()

        ## now test to make sure metadata is saved in database

        if pyfusion.orm_manager.IS_ACTIVE:
            session = pyfusion.orm_manager.Session()
            some_fs = session.query(FlucStruc).all().pop()
            self.assertEqual(some_fs.meta, multichannel_data.meta)





class CheckSciPyFilters(PfTestBase):
    def test_sp_filter_butterworth_bandpass(self):
        n_ch = 3
        n_samples = 1024
        sample_period = 1.e-6
        # Let's generate a test signal with strong peaks at 20kHz and
        # 60kHz and a weaker peak at 40kHz. 
        
        mode_20kHz = {'amp': 10.0, 'freq': 20.0e3, 'mode_number':3, 'phase':0.2}
        mode_40kHz = {'amp':  2.0, 'freq': 40.0e3, 'mode_number':4, 'phase':0.3}
        mode_60kHz = {'amp': 10.0, 'freq': 60.0e3, 'mode_number':5, 'phase':0.4}

        multichannel_data = get_multimode_test_data(channels=get_n_channels(n_ch),
                                                    timebase = Timebase(np.arange(n_samples)*sample_period),
                                                    modes = [mode_20kHz, mode_40kHz, mode_60kHz],
                                                    noise = 0.1)

        filtered_data = multichannel_data.sp_filter_butterworth_bandpass([35.e3, 45.e3], [25.e3, 55.e3], 1.0, 10.0)


class CheckFlucstrucPhases(PfTestBase):
    """Test code to replicate bug found by Shaun

    Pyfusion uses different syntax depending if sql is enabled.
    """

    def test_flucstruc_phases(PfTestCase):
        
        n_ch = 10
        n_samples = 5000
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)

        data_reduced_time=multichannel_data.reduce_time([0,0.002]).subtract_mean().normalise(method='v',separate=True)

        fs_set=data_reduced_time.flucstruc()
        phases = []
        for fs in fs_set:
            for j in range(0,len(fs.dphase)):
                phases.append(fs.dphase[j].delta)



class CheckFilterCopy(PfTestBase):
    """Check that by default, data filters alter a copy of the input data object not the object itself."""

    def test_timeseries_filter_copy(self):
        # Use reduce_time filter for testing...
        n_ch = 10
        n_samples = 5000
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)
        new_data = multichannel_data.reduce_time([0,1.e-3])
        self.assertFalse(new_data is multichannel_data)

    def test_timeseries_filter_nocopy(self):
        # Use reduce_time filter for testing...
        n_ch = 10
        n_samples = 5000
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)
        new_data = multichannel_data.reduce_time([0,1.e-3], copy=False)
        self.assertTrue(new_data is multichannel_data)


    def test_dataset_filter_copy(self):
        
        n_ch = 10
        n_samples = 640
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)
        dataset = multichannel_data.segment(64)
        new_dataset = dataset.segment(16)

    def test_dataset_filter_nocopy(self):
        n_ch = 10
        n_samples = 640
        timebase = Timebase(np.arange(n_samples)*1.e-6)
        channels = ChannelList(*(Channel('ch_%d' %i, Coords('cylindrical',(1.0,i,0.0))) for i in 2*np.pi*np.arange(n_ch)/n_ch))
        multichannel_data = get_multimode_test_data(channels = channels,
                                                    timebase = timebase,
                                                    noise = 0.5)
        dataset = multichannel_data.segment(64, copy=False)
        new_dataset = dataset.segment(16, copy=False)


CheckFilterCopy.dev = False
