"""Acquisition module for generating fake timeseries data for testing purposes.

At present, only a single channel sine wave generator is provided. Available configuration parameters are:

=================== ===========
parameter           description
=================== ===========
:attr:`t0`          Starting time of signal timebase.
:attr:`n_samples`   Number of samples.
:attr:`sample_freq` Sample frequency (Hz).
:attr:`frequency`   Frequency of test sine-wave signal (Hz).
:attr:`amplitude`   Amplitude of test sine-wave signal.
=================== ===========

All parameters are required.

For example, with the following configuration::

 [Acquisition:fake_acq]
 acq_class = pyfusion.acquisition.FakeData.acq.FakeDataAcquisition
 
 [Diagnostic:fake_data]
 data_fetcher = pyfusion.acquisition.FakeData.fetch.SingleChannelSineFetcher
 t0 = 0.0
 n_samples = 1024
 sample_freq = 1.e6
 frequency = 2.e4
 amplitude = 2.5

we can generate a 20 kHz sine wave::

 >>> import pyfusion as pf
 >>> shot = 12345
 >>> acq = pf.getAcquisition("fake_acq")
 >>> data = acq.getdata(shot, "fake_data")
 >>> data.timebase
 Timebase([  0.00000000e+00,   1.00000000e-06,   2.00000000e-06, ...,
          1.02100000e-03,   1.02200000e-03,   1.02300000e-03])
 >>> data.signal
 Signal([ 0.        ,  0.31333308,  0.62172472, ...,  1.20438419,
         0.92031138,  0.62172472])


"""


