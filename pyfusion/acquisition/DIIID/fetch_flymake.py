"""Subclass of MDSplus data fetcher to grab additional H1-specific metadata."""
import pyfusion, os
from pyfusion.acquisition.MDSPlus.fetch import MDSPlusDataFetcher, get_tree_path
import pyfusion.acquisition.MDSPlus.h1ds as mdsweb
from pyfusion.acquisition.base import MultiChannelFetcher
from pyfusion.data.timeseries import TimeseriesData, Signal, Timebase
from pyfusion.data.base import Coords, Channel, ChannelList, \
    get_coords_for_channel
import traceback
import numpy as np
from pyfusion.conf.utils import import_setting, kwarg_config_handler, \
     get_config_as_dict, import_from_str

class DIIIDDataFetcherPTdata(MDSPlusDataFetcher):
    """Subclass of MDSplus fetcher to get additional H1-specific metadata."""
    def setup(self):
        pass

    def do_fetch(self):
        print(self.pointname)
        print(self.shot)
        if self.NC!=None:
            print(self.NC)
            t_name = '{}_time'.format(self.pointname)
            NC_vars = self.NC.variables.keys()
            if self.pointname in NC_vars:
                print('Reading cache!!!!')
                t_axis = self.NC.variables[t_name].data[:].copy()
                data = self.NC.variables[self.pointname].data[:].copy()
        else:
            tmp = self.acq.connection.get('ptdata2("{}",{})'.format(self.pointname, self.shot))
            data = tmp.data()
            tmp = self.acq.connection.get('dim_of(ptdata2("{}",{}))'.format(self.pointname, self.shot))
            t_axis = tmp.data()
            self.write_cache = True
        print(t_axis)
        print(data)
        coords = get_coords_for_channel(**self.__dict__)
        ch = Channel(self.pointname, coords)
        # con=MDS.Connection('atlas.gat.com::')
        # pointname = 'MPI66M067D'
        # shot = 164950
        # tmp = con.get('ptdata2("{}",{})'.format(pointname, shot))
        # dat = tmp.data()
        # tmp = con.get('dim_of(ptdata2("{}",{}))'.format(pointname, shot))
        # t = tmp.data()
        if self.NC!=None and self.write_cache:
            print self.pointname
            self.NC.createDimension(t_name, len(t_axis))
            f_time = self.NC.createVariable(t_name,'d',(t_name,))
            f_time[:] = +t_axis
            print('Wrote time')
            sig = self.NC.createVariable(self.pointname,'f',(t_name,))
            sig[:] = +data
            print('Wrote signal')
        output_data = TimeseriesData(timebase=Timebase(t_axis),
                                signal=Signal(data), channels=ch)
        # output_data = super(DIIIDDataFetcherPTdata, self).do_fetch()
        # coords = get_coords_for_channel(**self.__dict__)
        # ch = Channel(self.mds_path, coords)
        # output_data.channels = ch
        # output_data.meta.update({'shot':self.shot, 'kh':self.get_kh()})
        # print(ch)
        output_data.config_name = ch
        self.fetch_mode = 'ptdata'
        return output_data

class DIIIDMultiChannelFetcher(MultiChannelFetcher):
    """Fetch data from a diagnostic with multiple timeseries channels.

    This fetcher requres a multichannel configuration section such as::

     [Diagnostic:H1_mirnov_array_1]
     data_fetcher = pyfusion.acquisition.base.MultiChannelFetcher
     channel_1 = H1_mirnov_array_1_coil_1
     channel_2 = H1_mirnov_array_1_coil_2
     channel_3 = H1_mirnov_array_1_coil_3
     channel_4 = H1_mirnov_array_1_coil_4

    The channel  names must be  `channel\_` followed by an  integer, and
    the channel  values must correspond to  other configuration sections
    (for        example       ``[Diagnostic:H1_mirnov_array_1_coil_1]``,
    ``[Diagnostic:H1_mirnov_array_1_coil_1]``, etc)  which each return a
    single               channel               instance               of
    :py:class:`~pyfusion.data.timeseries.TimeseriesData`.
    """
    def fetch(self, interp_if_diff = True):
        """Fetch each  channel and combine into  a multichannel instance
        of :py:class:`~pyfusion.data.timeseries.TimeseriesData`.

        :rtype: :py:class:`~pyfusion.data.timeseries.TimeseriesData`
        """
        print('******** hello world ***********')
        ## initially, assume only single channel signals
        ordered_channel_names = self.ordered_channel_names()
        data_list = []
        channels = ChannelList()
        timebase = None
        meta_dict={}
        from scipy.io import netcdf
        fname = '/u/haskeysr/tmp/{}.nc'.format(self.shot)
        write_cache=False; read_cache=False
        if os.path.exists(fname):
            NC = netcdf.netcdf_file(fname,'a',version=2)
        else:
            NC = netcdf.netcdf_file(fname,'w',version=2)
        for chan in ordered_channel_names:
            fetcher_class = import_setting('Diagnostic', chan, 'data_fetcher')
            tmp_data = fetcher_class(self.acq, self.shot,
                                     config_name=chan, NC=NC).fetch()
            channels.append(tmp_data.channels)
            meta_dict.update(tmp_data.meta)
            if timebase == None:
                timebase = tmp_data.timebase
                data_list.append(tmp_data.signal)
            else:
                try:
                    assert_array_almost_equal(timebase, tmp_data.timebase)
                    data_list.append(tmp_data.signal)
                except:
                    if interp_if_diff:
                        data_list.append(np.interp(timebase, tmp_data.timebase, tmp_data.signal))
                    else:
                        raise
        
        NC.close()
        signal=Signal(data_list)
        output_data = TimeseriesData(signal=signal, timebase=timebase,
                                     channels=channels)
        #output_data.meta.update({'shot':self.shot})
        output_data.meta.update(meta_dict)
        return output_data

