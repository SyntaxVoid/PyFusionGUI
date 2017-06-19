"""Data fetcher class for delimiter-separated value (DSV) data."""

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.timeseries import Signal, Timebase, TimeseriesData
from pyfusion.data.base import Coords, Channel, ChannelList

from numpy import genfromtxt

# Generate generic channel with dummy coordinates.
generic_ch = lambda x: Channel('channel_%03d' %(x+1), Coords('dummy', (x,0,0)))

class DSVMultiChannelTimeseriesFetcher(BaseDataFetcher):
    """Fetch DSV data from specified filename.

    
    This data fetcher uses two configuration parameters, filename (required) and delimiter (optioanl).

    The filename parameter can include a substitution string ``(shot)`` which will be replaced with the shot number.

    By default, whitespace is used for the delimiter character (if the delimiter parameter is not provided.)
    """
    def do_fetch(self):
        delimiter = self.__dict__.get("delimiter", None)
        data = genfromtxt(self.filename.replace("(shot)", str(self.shot)),
                          unpack=True, delimiter=delimiter)        

        # len(data) is number of channels + 1 (timebase)
        n_channels = len(data)-1

        ch_generator = (generic_ch(i) for i in range(n_channels))
        ch = ChannelList(*ch_generator)

        return TimeseriesData(timebase=Timebase(data[0]),
                              signal=Signal(data[1:]),
                              channels=ch)
