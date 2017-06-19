"""LHD data fetchers.
Large chunks of code copied from Boyd, not covered by unit tests,
then copied back to this version to read .npz local data.  THis is different
to reading the files obtained by retrieve
"""

def newload(filename, verbose=1):
    """ Intended to replace load() in numpy
    """
    from numpy import load as loadz
    from numpy import cumsum
    dic=loadz(filename)
#    if dic['version'] != None:
#    if len((dic.files=='version').nonzero())>0:
    if len(dic.files)>3:
        if verbose>2: print ("local v%d " % (dic['version'])),
    else: 
        if verbose>2: print("local v0: simple "),
        return(dic)  # quick, minimal return

    if verbose>2: print(' contains %s' % dic.files)
    signalexpr=dic['signalexpr']
    timebaseexpr=dic['timebaseexpr']
# savez saves ARRAYS always, so have to turn array back into scalar    
    exec(signalexpr.tolist())
    exec(timebaseexpr.tolist())
    return({"signal":signal, "timebase":timebase, "parent_element": dic['parent_element']})

from os import path
from numpy import mean, array, double, arange, dtype
import numpy as np
import array as Array
import pyfusion as pf

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList, get_coords_for_channel

VERBOSE = 1
#data_filename = "%(diag_name)s-%(shot)d-1-%(channel_number)s"
data_filename = "%(shot)d_%(diag_name)s.npz"

class LHDBaseDataFetcher(BaseDataFetcher):
    pass

class LHDTimeseriesDataFetcher(LHDBaseDataFetcher):

#        chnl = int(self.channel_number)
#        dggn = self.diag_name

    def do_fetch(self):
        chan_name = (self.diag_name.split('-'))[-1]  # remove -
        filename_dict = {'diag_name':chan_name, 
                         'shot':self.shot}

        self.basename = path.join(pf.config.get('global', 'localdatapath'), data_filename %filename_dict)
    
        files_exist = path.exists(self.basename)
        if not files_exist:
            raise Exception, "file " + self.basename + " not found."
        else:
            signal_dict = newload(self.basename)
            
        if ((chan_name == array(['MP5','HMP13','HMP05'])).any()):  flip = -1.
        else: flip = 1.
        if self.diag_name[0]=='-': flip = -flip
#        coords = get_coords_for_channel(**self.__dict__)
        ch = Channel(self.diag_name,  Coords('dummy', (0,0,0)))
        output_data = TimeseriesData(timebase=Timebase(signal_dict['timebase']),
                                 signal=Signal(flip*signal_dict['signal']), channels=ch)
        output_data.meta.update({'shot':self.shot})

        return output_data

