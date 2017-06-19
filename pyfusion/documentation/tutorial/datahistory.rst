.. _tut-history:

*******************************************
What have I done?? -- Checking data history
*******************************************

Any pyfusion object can be considered as the result of a series of operations performed on some original data sourced from a data acquisition system. It is easy to lose track of which filter settings, etc have been applied to any data - pyfusion takes care of that for you. For example::

 >>> from numpy.random import rand

 >>> from pyfusion.data.base import Coords
 >>> from pyfusion.data.timeseries import TimeseriesData, generate_timebase, Signal

 >>> # generate test timebase
 >>> tb = generate_timebase(t0=-0.5, n_samples=1.e2, sample_freq=1.e2)
 >>> # generate random signal
 >>> tsd = TimeseriesData(timebase=tb,
 ...                      signal=Signal(rand(len(tb))),
 ...                      coords=[Coords()])

 >>> # apply some filters
 >>> tsd.subtract_mean()
 >>> tsd.normalise(method='rms')

 >>> # have a look at what we've done...
 >>> print tsd.history
 2010-06-01 00:13:01.612530 > New TimeseriesData
 2010-06-01 00:13:01.612628 > subtract_mean()
 2010-06-01 00:13:01.612895 > normalise(method='rms')


The data log is automatically provided for all filters which are added via the ``@register()`` decorator. Currently, no useful information is provided about how the original data was created (which shot, which channel, etc) -- this will be added soon. 
