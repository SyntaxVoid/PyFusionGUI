"""MDSPlus data fetchers. """

import os, re
import numpy as np
import MDSplus
from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, ChannelList, Channel
import pyfusion.acquisition.MDSPlus.h1ds as mdsweb

mds_path_regex = re.compile(
    r'^\\(?P<tree>\w+?)::(?P<tagname>\w+?)[.|:](?P<nodepath>[\w.:]+)')

def get_tree_path(path_string):
     """Use regex to extract mdsplus tree, tag and node from full path"""
     components = mds_path_regex.search(path_string)
     ret_dict = {'tree':components.group('tree'),
                 'tagname':components.group('tagname'),
                 'nodepath':components.group('nodepath')}
     return ret_dict

def get_tsd_from_node(fetcher, node):
     """Return pyfusion TimeSeriesData corresponding to an MDSplus signal node."""
     # TODO: load actual coordinates
     ch = Channel(fetcher.mds_path_components['nodepath'],
                  Coords('dummy', (0,0,0)))
     signal = Signal(node.data())    
     dim = node.dim_of().data()
     # TODO: stupid hack,  the test signal has dim  of [[...]], real data
     # has [...].  Figure out  why. (...probably because  original signal
     # uses a build_signal function)
     if len(dim) == 1:
          dim = dim[0]
     timebase = Timebase(dim)
     output_data = TimeseriesData(timebase=timebase, signal=signal, channels=ch)
     output_data.meta.update({'shot':fetcher.shot})
     return output_data

class MDSPlusDataFetcher(BaseDataFetcher):
     """Determine which access mode should be used, and fetch the MDSplus data."""

     def setup(self):
          self.mds_path_components = get_tree_path(self.mds_path)
          if hasattr(self.acq, '%s_path' %self.mds_path_components['tree']):
               self.tree = MDSplus.Tree(self.mds_path_components['tree'],
                                        self.shot)
               self.fetch_mode = 'path'
          elif self.acq.server_mode == 'mds':
               self.acq.connection.openTree(self.mds_path_components['tree'],
                                            self.shot)
               self.fetch_mode = 'thin client'
          elif self.acq.server_mode == 'http':
               self.fetch_mode = 'http'
          else:
               raise Exception('Cannot determine MDSPlus fetch mode')

     def do_fetch(self):
          # TODO support non-signal datatypes
          if self.fetch_mode == 'thin client':
               ch = Channel(self.mds_path_components['nodepath'],
                            Coords('dummy', (0,0,0)))
               data = self.acq.connection.get(self.mds_path_components['nodepath'])
               dim = self.acq.connection.get('dim_of(%s)' %self.mds_path_components['nodepath'])
               # TODO: fix this hack (same hack as when getting signal from node)
               if len(data.shape) > 1:
                    data = np.array(data)[0,]
               if len(dim.shape) > 1:
                    dim = np.array(dim)[0,]
               output_data = TimeseriesData(timebase=Timebase(dim),
                                            signal=Signal(data), channels=ch)
               output_data.meta.update({'shot':self.shot})
               return output_data

          elif self.fetch_mode == 'http':
               data_url = self.acq.server + '/'.join([self.mds_path_components['tree'],
                                                      str(self.shot),
                                                      self.mds_path_components['tagname'],
                                                      self.mds_path_components['nodepath']])
               
               data = mdsweb.data_from_url(data_url)
               ch = Channel(self.mds_path_components['nodepath'], Coords('dummy', (0,0,0)))
               t = Timebase(data.data.dim)
               s = Signal(data.data.signal)
               output_data = TimeseriesData(timebase=t, signal=s, channels=ch)
               output_data.meta.update({'shot':self.shot})
               return output_data

          else:
               #print self.mds_path
               node = self.tree.getNode(self.mds_path)
               #print node.dtype
               #if int(node.dtype) == 195:
               if (int(node.dtype) == 195) or (int(node.dtype) == 192):#Shaun hack to work with PMT data
                    return get_tsd_from_node(self, node)
               else:
                    raise Exception('Unsupported MDSplus node type')

     def pulldown(self):
          if self.fetch_mode == 'thin client':
               self.acq.connection.closeTree(self.mds_path_components['tree'], self.shot)
