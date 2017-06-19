"""Timeseries data classes."""

from datetime import datetime
import numpy as np

from pyfusion.data.base import BaseData, BaseOrderedDataSet, FloatDelta
from pyfusion.data.utils import cps, peak_freq, remap_periodic, list2bin, bin2list
from pyfusion.data.base import PfMetaData
import pyfusion
from pyfusion.orm.utils import orm_register

import scipy

try:
    from sqlalchemy import Table, Column, Integer, ForeignKey, Float
    from sqlalchemy.orm import mapper, relation
except ImportError:
    # TODO: Use logger!
    print "cannot load sqlalchemy"
        
class Timebase(np.ndarray):
    """Timebase vector with parameterised internal representation.

    see doc/subclassing.py in numpy code for details on subclassing ndarray
    """
    def __new__(cls, input_array):
        # should this follow the example in doc/subclassing.py?... (it doesn't)
        obj = np.asarray(input_array).view(cls).copy()
        obj.sample_freq = 1.0/(obj[1]-obj[0])
        obj.meta = PfMetaData()
        return obj

    def is_contiguous(self):
        return max(((self[1:]-self[:-1])-1.0/self.sample_freq)**2) < (0.1/self.sample_freq)**2

    def normalise_freq(self, input_freq):
        """Normalise input frequencies to [0,1] where 1 is pi*sample_freq"""
        try:
            return input_freq/(0.5*self.sample_freq)
        except:
            # sample_freq should maybe be self.sample_freq? i don't remember
            return [i/(0.5*self.sample_freq) for i in sample_freq]

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.sample_freq = getattr(obj, 'sample_freq', None)
        self.meta = getattr(obj, 'meta', None)
        # We do not need to return anything
        


def generate_timebase(t0=0.0, n_samples=1.e4, sample_freq=1.e6):
    sample_time = 1./sample_freq
    return Timebase(np.arange(t0, t0+n_samples*sample_time, sample_time))
"""
class Timebase:
    def __init__(self, t0=None, n_samples=None, sample_freq=None):
        self.t0 = t0
        self.n_samples = n_samples
        self.sample_freq = sample_freq
        self.timebase = self.generate_timebase()
    def generate_timebase(self):
"""

class Signal(np.ndarray):
    """Timeseries signal class with (not-yet-implemented) configurable
    digitisation.

    see doc/subclassing.py in numpy code for details on subclassing ndarray
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).copy()
        obj.meta = PfMetaData()
        return obj

    def __array_finalize__(self,obj):
        self.meta = getattr(obj, 'meta', None)


    def n_channels(self):
        if self.ndim == 1:
            return 1
        else:
            return self.shape[0]

    def n_samples(self):
        if self.ndim == 1:
            return self.shape[0]
        else:
            return self.shape[1]

    def get_channel(self, channel_number):
        """allows us to use get_channel(0) no matter what ndim is"""
        if self.ndim == 1 and channel_number == 0:
            return self
        elif self.ndim > 1:
            return self[channel_number,:]
        else:
            raise ValueError

        
    #def add_channel(self, additional_array):
    ##### -- problems with resizing subclass of ndarray, need to be careful 
    #    additional_signal = Signal(additional_array)
    #    new_n_channels = self.n_channels() + additional_signal.n_channels()
    #    self.resize((new_n_channels, self.n_samples()))
    #    self[-additional_signal.n_channels():,:] = additional_signal
    

class TimeseriesData(BaseData):
    def __init__(self, timebase = None, signal=None, channels=None, bypass_length_check=False, **kwargs):
        self.timebase = timebase
        self.channels = channels
        self.scales = np.array([[1]])  # retain amplitude scaling info for later
        if bypass_length_check == True:
            self.signal = signal
        else:
            if signal.n_samples() == len(timebase):
                self.signal = signal
            else:
                raise ValueError, "signal has different number of samples to timebase"
        super(TimeseriesData, self).__init__(**kwargs)


    def generate_frequency_series(self, NFFT, step, window='hamming'):
        w =scipy.hamming(NFFT)
        if len(self.signal.shape)==2:
            signal = np.array([np.fft.rfft(w*self.signal[:,i:i+NFFT]/NFFT) for i in range(0, self.signal.shape[1]-NFFT, step)])
        else:
            signal = np.array([np.fft.rfft(w*self.signal[i:i+NFFT]/NFFT) for i in range(0, self.signal.shape[1]-NFFT, step)])
        timebase = np.array([np.average(self.timebase[i:i+NFFT]) 
                             for i in range(0, self.signal.shape[1]-NFFT, step)])

        #This is borrowed from here:
        #http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.fft.rfftfreq.html
        d = (self.timebase[1] - self.timebase[0])
        val = 1.0/(NFFT*d)
        N = NFFT//2 + 1
        frequency_base = np.round((np.arange(0, N, dtype=int)) * val,4)
        return FrequencyseriesData(frequency_base = frequency_base, timebase=timebase, 
                                   signal=signal,channels=self.channels,NFFT=NFFT, window='hamming',step=step)


@orm_register()
def orm_load_timeseries_data(man):
    man.tsd_table = Table('timeseriesdata', man.metadata,
                          Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
    #man.metadata.create_all()
    mapper(TimeseriesData, man.tsd_table, inherits=BaseData, polymorphic_identity='tsd')



class FrequencyseriesData(BaseData):
    def __init__(self, timebase = None, frequency_base = None, signal=None, channels=None, NFFT=None,window=None,step=None,**kwargs):
        self.timebase = timebase
        self.frequency_base = frequency_base
        self.channels = channels
        self.NFFT = NFFT
        self.window = window
        self.step = step
        self.signal = signal
        super(FrequencyseriesData, self).__init__(**kwargs)


@orm_register()
def orm_load_frequencyseries_data(man):
    man.tsd_table = Table('frequencyseriesdata', man.metadata,
                          Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
    #man.metadata.create_all()
    mapper(FrequencyseriesData, man.tsd_table, inherits=BaseData, polymorphic_identity='fsd')


class SVDData(BaseData):
    def __init__(self, chrono_labels, topo_channels, svd_input):
        """
        svd_input is a tuple as outputted by numpy.linalg.svd(data, 0)
        """
        self.channels = topo_channels
        self.chrono_labels = chrono_labels
        self.topos = np.transpose(svd_input[0])
        self.svs = svd_input[1]
        self.chronos = svd_input[2]
        self.E = sum(self.svs*self.svs)
        self.p = self.svs**2/self.E
        self.H = float((-1./np.log(len(self.svs)))*sum(self.p*np.log(self.p)))
        super(SVDData, self).__init__()

        
    def self_cps(self):
        try:
            return self._self_cps
        except AttributeError:
            self._self_cps = np.array([np.mean(cps(i,i)) for i in self.chronos])
            return self._self_cps


@orm_register()
def orm_load_svd_data(man):
    man.svd_table = Table('svddata', man.metadata,
                      Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True))
    #man.metadata.create_all()
    mapper(SVDData, man.svd_table, inherits=BaseData, polymorphic_identity='svddata')


class FlucStruc(BaseData):
    def __init__(self, svd_data, sv_list, timebase, min_dphase = -np.pi):
        # NOTE I'd prefer not to duplicate info here which is in svd_data - should be able to refer to that, once sqlalchemy is hooked in
        #self.topo_channels = svd_data.topo_channels
        self.channels = svd_data.channels
        #self.svs = sv_list
        if len(sv_list) == 1:
            self.a12 = 0
        else:
            # this ratio was inverted accidentally at first
            self.a12 = svd_data.svs[sv_list[1]]/svd_data.svs[sv_list[0]]
        self._binary_svs = list2bin(sv_list)
        # peak frequency for fluctuation structure
        self.timebase = timebase
        self.freq, self.freq_elmt = peak_freq(svd_data.chronos[sv_list[0]], self.timebase)
        self.t0 = timebase[0]
        # singular value filtered signals
        self.signal = np.dot(np.transpose(svd_data.topos[sv_list,:]),
                           np.dot(np.diag(svd_data.svs.take(sv_list)), svd_data.chronos[sv_list,:]))
        # phase differences between nearest neighbour channels
        self.dphase, self.fourier_values = self._get_dphase(min_dphase=min_dphase, get_fourier_value = 1)
        self.p = np.sum(svd_data.svs.take(sv_list)**2)/svd_data.E
        self.H = svd_data.H
        self.E = svd_data.E
        super(FlucStruc, self).__init__()

    def save(self):
        self.dphase.save()
        super(FlucStruc, self).save()

    def svs(self):
        return bin2list(self._binary_svs)

    def _get_dphase(self, min_dphase = -np.pi, get_fourier_value=0):
        """
        remap to [min_dphase, min_dphase+2pi]
        #SRH modified 24June2013 to return the complex value also
        """
        if get_fourier_value==0:
            phases = np.array([self._get_single_channel_phase(i) for i in range(self.signal.shape[0])])
        else:
            fourier_vals = []; phases = []
            for i in range(self.signal.shape[0]):
                phases_tmp, fourier_vals_tmp = self._get_single_channel_phase(i, get_fourier_value = 1)
                phases.append(phases_tmp)
                fourier_vals.append(fourier_vals_tmp)
            phases = np.array(phases)
            fourier_vals = np.array(fourier_vals)
        d_phases = remap_periodic(phases[1:]-phases[:-1], min_val = min_dphase)
        #d_phase_dataset = OrderedDataSet(ordered_by="channel_1.name")
        d_phase_dataset = BaseOrderedDataSet('d_phase_%s' %datetime.now())
        ## append then sort should be faster than ordereddataset.add() [ fewer sorts()]
        for i, d_ph in enumerate(d_phases):
            d_phase_dataset.append(FloatDelta(self.channels[i], self.channels[i+1], d_ph))

        #d_phase_dataset.sort()
        if get_fourier_value==0:
            return d_phase_dataset
        else:
            return d_phase_dataset, fourier_vals


    def _get_single_channel_phase(self, ch_id, get_fourier_value = 0):
        #SRH modified 24June2013 to return the complex value also
        data_fft = np.fft.fft(self.signal[ch_id])
        d_val = data_fft[self.freq_elmt]
        phase_val = np.arctan2(d_val.imag,d_val.real)
        if get_fourier_value == 0:
            return phase_val
        else:
            return phase_val, d_val

@orm_register()
def orm_load_flucstrucs(man):
    man.flucstruc_table = Table('flucstrucs', man.metadata,
                            Column('basedata_id', Integer, ForeignKey('basedata.basedata_id'), primary_key=True),
                            Column('_binary_svs', Integer),
                            Column('freq', Float),
                            Column('a12', Float),
                            Column('t0', Float),    
                            Column('p', Float),    
                            Column('H', Float),    
                            Column('dphase_id', Integer, ForeignKey('baseordereddataset.id'), nullable=False))    
    man.metadata.create_all()
    mapper(FlucStruc, man.flucstruc_table, inherits=BaseData,
           polymorphic_identity='flucstruc', properties={'dphase': relation(BaseOrderedDataSet)})
