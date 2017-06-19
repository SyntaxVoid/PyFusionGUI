"""
Some un-pythonic code here (checking instance type inside
function). Need to figure out a better way to do this.
"""
from datetime import datetime
import copy
from numpy import searchsorted, arange, mean, resize, repeat, fft, conjugate, linalg, array, zeros_like, take, argmin, pi, cumsum
from numpy import correlate as numpy_correlate
import numpy as np
#from pyfusion.data.timeseries import Signal
#from pyfusion.data.base import PfMetaData
try:
    from scipy import signal as sp_signal
except:
    # should send message to log...
    pass

import pyfusion




def cps(a,b):
    return fft.fft(a)*conjugate(fft.fft(b))


filter_reg = {}


def register(*class_names):
    def reg_item(filter_method):
        for cl_name in class_names:
            if not filter_reg.has_key(cl_name):
                filter_reg[cl_name] = [filter_method]
            else:
                filter_reg[cl_name].append(filter_method)
        return filter_method
    return reg_item

"""
class MetaFilter(type):
    def __new__(cls, name, bases, attrs):
        filter_methods = filter_reg.get(name, [])
        attrs.update((i.__name__,i) for i in filter_methods)
        return super(MetaFilter, cls).__new__(cls, name, bases, attrs)
"""

@register("TimeseriesData", "DataSet")
def reduce_time(input_data, new_time_range):
    from pyfusion.data.base import DataSet
    if isinstance(input_data, DataSet):
        #output_dataset = input_data.copy()
        #output_dataset.clear()
        output_dataset = DataSet(input_data.label+'_reduce_time')
        for data in input_data:
            try:
                output_dataset.append(data.reduce_time(new_time_range))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'reduce_time' not applied to item in dataset")
        return output_dataset

    new_time_args = searchsorted(input_data.timebase, new_time_range)
    input_data.timebase =input_data.timebase[new_time_args[0]:new_time_args[1]]
    if input_data.signal.ndim == 1:
        input_data.signal = input_data.signal[new_time_args[0]:new_time_args[1]]
    else:
        input_data.signal = input_data.signal[:,new_time_args[0]:new_time_args[1]]
    return input_data


@register("TimeseriesData", "DataSet")
def segment(input_data, n_samples, overlap=1.0, datalist= 0):
    """Break into segments length n_samples.

    Overlap of 2.0 starts a new segment halfway into previous, overlap=1 is
    no overlap.  overlap should divide into n_samples.  Probably should
    consider a nicer definition such as in pyfusion 0

    if datalist = 0 returns a DataSet object otherwise, returns a OrderedDataSet object
    """
    from pyfusion.data.base import DataSet, OrderedDataSet
    from pyfusion.data.timeseries import TimeseriesData
    if isinstance(input_data, DataSet):
        output_dataset = DataSet()
        for ii,data in enumerate(input_data):
            try:
                output_dataset.update(data.segment(n_samples))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'segment' not applied to item in dataset")
        return output_dataset

    #SH modification incase ordering is important... i.e you are doing 
    #two processing two different arrays at the same time (in different Timeseries objects)
    #and you don't want to loose the time relationship between them
    if datalist:
        output_data = OrderedDataSet('segmented_%s, %d samples, %.3f overlap' %(datetime.now(), n_samples, overlap))
    else:
        output_data = DataSet('segmented_%s, %d samples, %.3f overlap' %(datetime.now(), n_samples, overlap))
    #SH : 24May2013 fixed bug here - before, the index was allowed to go past 
    #the length of samples, gives smalled length data towards the end - fixed to finish the
    #last time we can get n_samples length

    #for el in arange(0,len(input_data.timebase), n_samples/overlap):
    for el in arange(0,len(input_data.timebase) - n_samples, n_samples/overlap):
        if input_data.signal.ndim == 1:
            tmp_data = TimeseriesData(timebase=input_data.timebase[el:el+n_samples],
                                      signal=input_data.signal[el:el+n_samples],
                                      channels=input_data.channels, bypass_length_check=True)
        else:
            tmp_data = TimeseriesData(timebase=input_data.timebase[el:el+n_samples],
                                      signal=input_data.signal[:,el:el+n_samples],
                                      channels=input_data.channels, bypass_length_check=True)
            
        tmp_data.meta = input_data.meta.copy()
        if datalist:
            output_data.append(tmp_data)
        else:
            output_data.add(tmp_data)
    return output_data

@register("DataSet")
def remove_noncontiguous(input_dataset):
    remove_list = []
    for item in input_dataset:
        if not item.timebase.is_contiguous():
            remove_list.append(item)
    for item in remove_list:
        input_dataset.remove(item)
    return input_dataset

@register("TimeseriesData", "DataSet")
def normalise(input_data, method='peak', separate=False):
    from numpy import mean, sqrt, max, abs, var, atleast_2d
    from pyfusion.data.base import DataSet
    # this allows method='0'(or 0) to prevent normalisation for cleaner code
    # elsewhere
    if (method == 0) or (method == '0'): return(input_data)
    if isinstance(input_data, DataSet):
        output_dataset = DataSet(input_data.label+"_normalise")
        for d in input_data:
            output_dataset.add(normalise(d, method=method, separate=separate))
        return output_dataset
    if method.lower() in ['rms', 'r']:
        if input_data.signal.ndim == 1:
            norm_value = sqrt(mean(input_data.signal**2))
        else:
            rms_vals = sqrt(mean(input_data.signal**2, axis=1))
            if separate == False:
                rms_vals = max(rms_vals)
            norm_value = atleast_2d(rms_vals).T            
    elif method.lower() in ['peak', 'p']:
        if input_data.signal.ndim == 1:
            norm_value = abs(input_data.signal).max(axis=0)
        else:
            max_vals = abs(input_data.signal).max(axis=1)
            if separate == False:
                max_vals = max(max_vals)
            norm_value = atleast_2d(max_vals).T
    elif method.lower() in ['var', 'variance', 'v']:
        if input_data.signal.ndim == 1:
            norm_value = var(input_data.signal)
        else:
            var_vals = var(input_data.signal, axis=1)
            if separate == False:
                var_vals = max(var_vals)
            norm_value = atleast_2d(var_vals).T            
    input_data.signal = input_data.signal / norm_value
    #print('norm_value = %s' % norm_value)
    input_data.scales = norm_value
    return input_data
    
@register("TimeseriesData")
def svd(input_data):
    from timeseries import SVDData
    return SVDData(input_data.timebase, input_data.channels, linalg.svd(input_data.signal, 0))


#@register("TimeseriesData", "SVDData")
def fs_group_geometric(input_data, max_energy = 1.0):
    """
    no filtering implemented yet
    we don't register this as a filter, because it doesn't return a Data or DataSet subclass
    TODO: write docs for how to use max_energy - not obvious if using flucstruc() filter...
    """
    from timeseries import SVDData
    #from base import OrderedDataSet

    if not isinstance(input_data, SVDData):
        input_data = input_data.subtract_mean().normalise(method="var").svd()

    output_fs_list = []#OrderedDataSet()

    if max_energy < 1.0:
        max_element = searchsorted(cumsum(input_data.p), max_energy)
        remaining_ids = range(max_element)
    else:
        remaining_ids = range(len(input_data.svs))
    
    self_cps = input_data.self_cps()

    while len(remaining_ids) > 1:
        rsv0 = remaining_ids[0]
        tmp_cp = [mean(abs(cps(input_data.chronos[rsv0], input_data.chronos[sv])))**2/(self_cps[rsv0]*self_cps[sv]) for sv in remaining_ids]
        tmp_cp_argsort = array(tmp_cp).argsort()[::-1]
        sort_cp = take(tmp_cp,tmp_cp_argsort)
        delta_cp = sort_cp[1:]-sort_cp[:-1]
        
        output_fs_list.append([remaining_ids[i] for i in tmp_cp_argsort[:argmin(delta_cp)+1]])
            

        for i in output_fs_list[-1]: remaining_ids.remove(i)
    if len(remaining_ids) == 1:
        output_fs_list.append(remaining_ids)

    return output_fs_list


#@register("SVDData")
def fs_group_threshold(input_data, threshold=0.2):
    """
    no filtering implemented yet
    we don't register this as a filter, because it doesn't return a Data or DataSet subclass
    """
    from timeseries import SVDData

    if not isinstance(input_data, SVDData):
        input_data = input_data.subtract_mean().normalise(method="var").svd()
    
    
    #svd_data = linalg.svd(norm_data.signal,0)
    output_fs_list = []

    #svs_norm_energy = array([i**2 for i in svd_data[1]])/input_data.E

    #max_element = searchsorted(cumsum(svs_norm_energy), energy_threshold)
    #remaining_ids = range(max_element)
    remaining_ids = range(len(input_data.svs))
    
    self_cps = input_data.self_cps()

    while len(remaining_ids) > 1:
        rsv0 = remaining_ids[0]
        tmp_cp = [mean(abs(cps(input_data.chronos[rsv0], input_data.chronos[sv])))**2/(self_cps[rsv0]*self_cps[sv]) for sv in remaining_ids]
        filtered_elements = [i for [i,val] in enumerate(tmp_cp) if val > threshold]
        output_fs_list.append([remaining_ids[i] for i in filtered_elements])
            

        for i in output_fs_list[-1]: remaining_ids.remove(i)
    if len(remaining_ids) == 1:
        output_fs_list.append(remaining_ids)

    return output_fs_list

@register("TimeseriesData")
def flucstruc(input_data, min_dphase = -pi, group=fs_group_geometric, method='rms', separate=True, label=None):
    from pyfusion.data.base import DataSet
    from pyfusion.data.timeseries import FlucStruc

    if label:
        fs_dataset = DataSet(label)
    else:
        fs_dataset = DataSet('flucstrucs_%s' %datetime.now())
    svd_data = input_data.subtract_mean().normalise(method, separate).svd()

    for fs_gr in group(svd_data):
        tmp = FlucStruc(svd_data, fs_gr, input_data.timebase, min_dphase=min_dphase)
        tmp.meta = input_data.meta
        fs_dataset.add(tmp)
    
    return fs_dataset


@register("TimeseriesData", "DataSet")
def subtract_mean(input_data):
    from pyfusion.data.base import DataSet
    if isinstance(input_data, DataSet):
        output_dataset = DataSet(input_data.label+"_subtract_mean")
        for d in input_data:
            output_dataset.add(subtract_mean(d))
        return output_dataset
    if input_data.signal.ndim == 1:
        mean_value = mean(input_data.signal)
    else:
        mean_vector = mean(input_data.signal, axis=1)
        mean_value = resize(repeat(mean_vector, input_data.signal.shape[1]), input_data.signal.shape)
    input_data.signal -= mean_value
    return input_data

###############################
## Wrappers to SciPy filters ##
###############################
@register("TimeseriesData")
def sp_filter_butterworth_bandpass(input_data, passband, stopband, max_passband_loss, min_stopband_attenuation):
    # The SciPy signal processing module uses normalised frequencies, so we need to normalise the input values
    norm_passband = input_data.timebase.normalise_freq(passband)
    norm_stopband = input_data.timebase.normalise_freq(stopband)
    ord,wn = sp_signal.filter_design.buttord(norm_passband, norm_stopband, max_passband_loss, min_stopband_attenuation)
    b, a = sp_signal.filter_design.butter(ord, wn, btype = 'bandpass')
    
    output_data = input_data

    for i,s in enumerate(output_data.signal):
        output_data.signal[i] = sp_signal.lfilter(b,a,s)

    return output_data


#########################################
## wrappers to numpy signal processing ##
#########################################
@register("TimeseriesData")
def correlate(input_data, index_1, index_2, **kwargs):
    return numpy_correlate(input_data.signal[index_1],
                           input_data.signal[index_2], **kwargs)


@register("TimeseriesData", "DataSet")
def change_time_base(input_data, new_time_base):
    '''New from SH....
    '''
    from pyfusion.data.base import DataSet
    from pyfusion.data.timeseries import Signal, Timebase
    if isinstance(input_data, DataSet):
        #output_dataset = input_data.copy()
        #output_dataset.clear()
        output_dataset = DataSet(input_data.label+'_new_time_base')
        for data in input_data:
            try:
                output_dataset.append(data.change_time_base(new_time_base))
            except AttributeError:
                pyfusion.logger.warning("Data filter 'change_time_base' not applied to item in dataset")
        return output_dataset

    #cut the signal and timebase matrices to the correct size
    new_data = copy.deepcopy(input_data)
    n_channels = input_data.signal.shape[0]
    new_data.signal = Signal(np.zeros((n_channels,new_time_base.shape[0]),dtype=np.float32))
    new_data.timebase = Timebase(new_time_base) 
    for i in range(input_data.signal.shape[0]):
        new_data.signal[i,:] = np.interp(new_time_base, input_data.timebase, input_data.signal[i,:])

    #if input_data.signal.ndim == 1:
    #    input_data.signal = input_data.signal[new_time_args[0]:new_time_args[1]]
    #else:
    #    input_data.signal = input_data.signal[:,new_time_args[0]:new_time_args[1]]
    return new_data
