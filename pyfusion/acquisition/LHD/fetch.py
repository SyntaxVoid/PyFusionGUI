"""LHD data fetchers.
Large chunks of code copied from Boyd, not covered by unit tests.
"""
import subprocess
import sys
import tempfile
from os import path
import array as Array
from numpy import mean, array, double, arange, dtype, load
import numpy as np

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList

this_dir = path.dirname(path.abspath(__file__)) 

VERBOSE = 1
data_filename = "%(diag_name)s-%(shot)d-1-%(channel_number)s"

class LHDBaseDataFetcher(BaseDataFetcher):
    pass

class LHDTimeseriesDataFetcher(LHDBaseDataFetcher):

    def do_fetch(self):
        # Allow for movement of Mirnov signals from A14 to PXI crate
        chnl = int(self.channel_number)
        dggn = self.diag_name
        if (dggn == 'FMD'):
            if (self.shot < 72380):
                dggn = 'SX8O'
                if chnl != 0: 
                    chnl = chnl + 33
                    if self.shot < 26208: chnl = chnl +1

        filename_dict = {'diag_name':dggn, 
                         'channel_number':chnl, 
                         'shot':self.shot}
        self.basename = path.join(self.filepath, data_filename %filename_dict)

        files_exist = path.exists(self.basename + ".dat") and path.exists(self.basename + ".prm")
        if not files_exist:
            if VERBOSE>3: print('RETR: retrieving %d chn %d to %s' % 
                              (self.shot, int(chnl),
                               self.filepath))
            tmp = retrieve_to_file(diagg_name=dggn, shot=self.shot, subshot=1, 
                                   channel=int(chnl), outdir = self.filepath)
            if not path.exists(self.basename + ".dat") and path.exists(self.basename + ".prm"):
                raise Exception, "something is buggered."

        return fetch_data_from_file(self)



zfile = load(path.join(this_dir,'a14_clock_div.npz'))

a14_clock_div = zfile['a14_clock_div']

def LHD_A14_clk(shot):
    """ Helper routine to fix up the undocumented clock speed changes in the A14"""

    """
    # The file a14_clock_div.npz replaces all this hard coded stuff
    # not sure about the exact turn over at 30240 and many others, not checked above 52k yet
    rate  = array([500,    1000,   500, 1000,    500,   250,  500,     250,   500,   250,   500,   250,   500,   250,   500])
    shots = array([26220, 30240, 30754, 31094, 31315, 49960,  51004, 51330, 51475, 51785, 52010, 52025, 52680, 52690, 52810, 999999])
    where_ge = (shot >= shots).nonzero()[0]
    if len(where_ge) < 1: 
        raise LookupError, 'a14_clock lookup: shot out of range'

    last_index = max(where_ge)
    rateHz = 1000.*rate[last_index]
    """
    div = a14_clock_div[shot]
    if div > 0: clk = 1e6/div
    else: clk = 0
    rateHz=clk
    # print(rateHz)
    return(rateHz)

def fetch_data_from_file(fetcher):
    prm_dict = read_prm_file(fetcher.basename+".prm")
    bytes = int(prm_dict['DataLength(byte)'][0])
    bits = int(prm_dict['Resolution(bit)'][0])
    if not(prm_dict.has_key('ImageType')):      #if so assume unsigned
        bytes_per_sample = 2
        dat_arr = Array.array('H')
        offset = 2**(bits-1)
        dtype = np.dtype('uint16')
    else:
        if prm_dict['ImageType'][0] == 'INT16':
            bytes_per_sample = 2
            if prm_dict['BinaryCoding'][0] == 'offset_binary':
                dat_arr = Array.array('H')
                offset = 2**(bits-1)
                dtype = np.dtype('uint16')
            elif prm_dict['BinaryCoding'][0] == "shifted_2's_complementary":
                dat_arr = Array.array('h')
                offset = 0
                dtype = np.dtype('int16')
            else: raise NotImplementedError,' binary coding ' + prm_dict['BinaryCoding']

    fp = open(fetcher.basename + '.dat', 'rb')
    dat_arr.fromfile(fp, bytes/bytes_per_sample)
    fp.close()

    clockHz = None

    if prm_dict.has_key('SamplingClock'): 
        clockHz =  double(prm_dict['SamplingClock'][0])
    if prm_dict.has_key('SamplingInterval'): 
        clockHz =  clockHz/double(prm_dict['SamplingInterval'][0])
    if prm_dict.has_key('ClockSpeed'): 
        if clockHz != None:
            pyfusion.utils.warn('Apparent duplication of clock speed information')
        clockHz =  double(prm_dict['ClockSpeed'][0])
        clockHz = LHD_A14_clk(fetcher.shot)  # see above
    if clockHz != None:
        timebase = arange(len(dat_arr))/clockHz
    else:  raise NotImplementedError, "timebase not recognised"
    
    ch = Channel("%s-%s" %(fetcher.diag_name, fetcher.channel_number), Coords('dummy', (0,0,0)))
    if fetcher.gain != None: 
        gain = fetcher.gain
    else: 
        gain = 1
    output_data = TimeseriesData(timebase=Timebase(timebase),
                                 signal=Signal(gain*dat_arr), channels=ch)
    output_data.meta.update({'shot':fetcher.shot})

    return output_data


def read_prm_file(filename):
    """ Read a prm file into a dictionary.  Main entry point is via filename,
    possibly reserve the option to access vai shot and subshot
    >>> pd = read_prm_file(filename=filename)
    >>> pd['Resolution(bit)']
    ['14', '2']
    """
    f = open(filename)
    prm_dict = {}
    for s in f:
        s = s.strip("\n")
        toks = s.split(',')  
        if len(toks)<2: print('bad line %s in %f' % (s, filename))
        key = toks[1]
        prm_dict.update({key: toks[2:]})
    f.close()
    return prm_dict

def retrieve_to_file(diagg_name=None, shot=None, subshot=None, 
                     channel=None, outdir = None, get_data=True):
    """ run the retrieve standalone program to get data to files,
    and/or extract the parameter and summary information.

    Retrieve Usage from Oct 2009 tar file:
    Retrieve DiagName ShotNo SubShotNo ChNo [FileName] [-f FrameNo] [-h TransdServer] [-p root] [-n port] [-w|--wait [-t timeout] ] [-R|--real ]
    """

    cmd = str("retrieve %s %d %d %d %s" % (diagg_name, shot, subshot, channel, path.join(outdir, diagg_name)))

    if (VERBOSE > 1): print('RETR: %s' % (cmd))
    retr_pipe = subprocess.Popen(cmd,  shell=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
    (resp,err) = retr_pipe.communicate()
    if (err != '') or (retr_pipe.returncode != 0):

        raise LookupError(str("Error %d accessing retrieve: cmd=%s\nstdout=%s, stderr=%s" % 
                              (retr_pipe.poll(), cmd, resp, err)))

    for lin in resp.split('\n'):
        if lin.find('parameter file')>=0:
            fileroot = lin.split('[')[1].split('.prm')[0]
    return(resp, err, fileroot)
