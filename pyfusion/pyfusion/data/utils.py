import os, string
import random as _random
from numpy import fft, conjugate, array, mean, arange, searchsorted, argsort, pi


try:
    import uuid
except: # python 2.4
    pass

## for python <=2.5 compat, bin() is only python >= 2.6
## code taken from http://stackoverflow.com/questions/1993834/how-change-int-to-binary-on-python-2-5
def __bin(value):
    binmap = {'0':'0000', '1':'0001', '2':'0010', '3':'0011',
              '4':'0100', '5':'0101', '6':'0110', '7':'0111',
              '8':'1000', '9':'1001', 'a':'1010', 'b':'1011',
              'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111'}
    if value == 0:
        return '0b0'
    
    return '0b'+''.join(binmap[x] for x in ('%x' % (value,))).lstrip('0') or '0'

try:
    _bin = bin
except NameError: # python <= 2.5
    _bin = __bin

def unique_id():
    try:
        return str(uuid.uuid4())
    except:
        return ''.join(_random.choice(string.letters) for i in range(50))


def cps(a,b):
    return fft.fft(a)*conjugate(fft.fft(b))

def peak_freq(signal,timebase,minfreq=0,maxfreq=1.e18):
    """
    TODO: old code: needs review
    this function only has a basic unittest to make sure it returns
    the correct freq in a simple case.
    """
    timebase = array(timebase)
    sig_fft = fft.fft(signal)
    sample_time = float(mean(timebase[1:]-timebase[:-1]))

    #SRH modification, frequencies seemed a little bit off because of the -1 in the denominator
    #Here we are trusting numpy....
    #fft_freqs = (1./sample_time)*arange(len(sig_fft)).astype(float)/(len(sig_fft)-1)
    fft_freqs = fft.fftfreq(len(sig_fft),d=sample_time)
    # only show up to nyquist freq
    new_len = len(sig_fft)/2
    sig_fft = sig_fft[:new_len]
    fft_freqs = fft_freqs[:new_len]
    [minfreq_elmt,maxfreq_elmt] = searchsorted(fft_freqs,[minfreq,maxfreq])
    sig_fft = sig_fft[minfreq_elmt:maxfreq_elmt]
    fft_freqs = fft_freqs[minfreq_elmt:maxfreq_elmt]
    
    peak_elmt = (argsort(abs(sig_fft)))[-1]
    return [fft_freqs[peak_elmt], peak_elmt]

def remap_periodic(input_array, min_val, period = 2*pi):
    while len(input_array[input_array<min_val]) > 0:
        input_array[input_array<min_val] += period
    while len(input_array[input_array>=min_val+period]) > 0:
        input_array[input_array>=min_val+period] -= period
    return input_array

def list2bin(input_list):
    # we explicitly cast to int(), as numpy's integer type clashes with sqlalchemy
    return int(sum(2**array(input_list)))

def bin2list(input_value):
    output_list = []
    bin_index_str = _bin(input_value)[2:][::-1]
    for ind,i in enumerate(bin_index_str):
        if i == '1':
            output_list.append(ind)
    return output_list

def split_names(names, pad=' '):
    """ Given an array of strings, return an array of the part of the string
    (e.g. channel name) that varies, and optionally the prefix and suffix.
    The array of varying parts is first in the tuple in case others are not
    wanted.  This is used to make the x labels of phase plots simpler and smaller.
    e.g.
    >>> split_names(['MP01','MP10'])
    (['01','10'], 'MP', '')
    """
    # make a new array with elements padded to the same length with <pad>
    nms = []
    maxlen = max([len(nm) for nm in names])
    for nm in names:
        nmarr = [c for c in nm]
        while len(nmarr)< maxlen: nmarr.append(pad)
        nms.append(nmarr)
    
    # the following numpy array comparisons look simple, but require the name string
    # to be exploded into chars.  Although a single string can be interchangeably 
    # referred to as a string or array of chars, these arrays they have to be 
    # re-constituted before return.
    #
    #    for nm in nms:     # for each nm
    #find the first mismatch - first will be the first char of the extracted arr
    nms_arr=array(nms)
    first=0
    while (first < maxlen and
           (nms_arr[:,first] == nms_arr[0,first]).all()):
        first += 1
    # and the last        
    last = maxlen-1
    while ((last >= 0) and
           (nms_arr[:,last] == nms_arr[0,last]).all()):
        last -= 1


    # check for no mismatch        
    if first==maxlen: return(['' for nm in names], ''.join(nms[0]),'')
    # otherwise return, (no need for special code for the case of no match at all)
    return(([''.join(s) for s in nms_arr[:,first:last+1]],
            ''.join(nms_arr[0,0:first]),
            ''.join(nms_arr[0,last+1:maxlen+1])))
