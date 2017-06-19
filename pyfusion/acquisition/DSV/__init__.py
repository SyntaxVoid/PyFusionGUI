"""Acquisition module for data in a delimiter-separated value (DSV) format.

================= ===========
parameter         description
================= ===========
:attr:`filename`  Name of data file, with ``(shot)`` substitution string, e.g. ``/data/(shot).dat`` -> ``/data/12345.dat`` for shot 12345. (required)
:attr:`delimiter` Delimiter character for values, e.g. ``,`` for comma separated value (CSV) format. (optional, default is whitespace)
================= ===========


This module provides support for reading data from a plain text file via
numpy's genfromtxt  function. The only  required configuration parameter
is  filename,  which  can  include  a shot  number  substitution  string
``(shot)``. An an example, consider the following datafile for 2-channel
timeseries signal for shot number 12345::

  # timebase   channel 1     channel 2
  3.000000e+00 -1.201389e-01  3.177084e-01
  3.000002e+00  6.437500e-01 -4.461806e-01
  3.000004e+00  5.347222e-02 -1.684028e-01
  3.000006e+00  1.923611e-01 -2.951390e-02
  3.000008e+00  4.006945e-01 -5.156250e-01
  3.000010e+00 -8.840278e-01  1.012153e+00
  3.000012e+00  2.618056e-01 -2.031250e-01
  3.000014e+00 -1.597222e-02 -1.336806e-01
  3.000016e+00 -1.597222e-02  1.788194e-01
  3.000018e+00  5.743055e-01 -7.586806e-01

If the  datafile is  saved at ``/data/mirnov_data_12345.txt``,  we could
use the following configuration file::

 [Acquisition:my_text_data]
 acq_class = pyfusion.acquisition.DSV.acq.DSVAcquisition
 
 [Diagnostic:mirnov_data]
 data_fetcher = pyfusion.acquisition.DSV.fetch.DSVMultiChannelTimeseriesFetcher
 filename = /data/mirnov_data_(shot).txt

And access the data with pyfusion::

 >>> import pyfusion as pf
 >>> acq = pf.getAcquisition("my_text_data")
 >>> data = acq.getdata(12345, "mirnov_data")
 >>> data.timebase
 Timebase([ 3.      ,  3.000002,  3.000004,  3.000006,  3.000008,  3.00001 ,
         3.000012,  3.000014,  3.000016,  3.000018])
 >>> data.signal[0]
 Signal([-0.1201389 ,  0.64375   ,  0.05347222,  0.1923611 ,  0.4006945 ,
         -0.8840278 ,  0.2618056 , -0.01597222, -0.01597222,  0.5743055 ])
 >>> data.signal[1]
 Signal([ 0.3177084, -0.4461806, -0.1684028, -0.0295139, -0.515625 ,
         1.012153 , -0.203125 , -0.1336806,  0.1788194, -0.7586806])


By  default,  pyfusion expects  values  to  be  delimited by  whitespace
characters.   The  delimiting  character   can  also   be  set   in  the
configuration   file,   for   example,   the  following   datafile   and
configuration give the same result as the above example::

  # timebase,   channel 1,     channel 2
  3.000000e+00, -1.201389e-01,  3.177084e-01
  3.000002e+00,  6.437500e-01, -4.461806e-01
  3.000004e+00,  5.347222e-02, -1.684028e-01
  3.000006e+00,  1.923611e-01, -2.951390e-02
  3.000008e+00,  4.006945e-01, -5.156250e-01
  3.000010e+00, -8.840278e-01,  1.012153e+00
  3.000012e+00,  2.618056e-01, -2.031250e-01
  3.000014e+00, -1.597222e-02, -1.336806e-01
  3.000016e+00, -1.597222e-02,  1.788194e-01
  3.000018e+00,  5.743055e-01, -7.586806e-01


where the configuration is::

 [Acquisition:my_text_data]
 acq_class = pyfusion.acquisition.DSV.acq.DSVAcquisition
 
 [Diagnostic:mirnov_data]
 data_fetcher = pyfusion.acquisition.DSV.fetch.DSVMultiChannelTimeseriesFetcher
 filename = /data/mirnov_data_(shot).txt
 delimiter = ,

Note that whitespace is stripped from configuration file values - if you
want to use  whitespace delimited data, as in  the first example, simply
omit the delimiter setting in your configuration.
"""
