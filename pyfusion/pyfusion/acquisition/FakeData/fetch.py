"""Fake data acquisition fetchers used for testing pyfusion code."""

from numpy import sin, pi

from pyfusion.acquisition.base import BaseDataFetcher
from pyfusion.data.base import Coords, ChannelList, Channel
from pyfusion.data.timeseries import TimeseriesData, Signal, generate_timebase

class SingleChannelSineFetcher(BaseDataFetcher):
    """Data fetcher for single channel sine wave."""
    def fetch(self):
        tb = generate_timebase(t0=float(self.t0), n_samples=int(self.n_samples),
                               sample_freq=float(self.sample_freq))
        sig = Signal(float(self.amplitude)*sin(2*pi*float(self.frequency)*tb))
        dummy_channel = Channel('ch_01',Coords('dummy', (0,0,0)))
        output_data = TimeseriesData(timebase=tb, signal=sig,
                                     channels=ChannelList(dummy_channel))
        output_data.meta.update({'shot':self.shot})
        return output_data
