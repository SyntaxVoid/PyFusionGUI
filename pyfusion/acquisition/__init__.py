"""The  acquisition package  is responsible  for fetching  data  from an
experimental database and returning pyfusion data objects.  Base classes
as well as datasystem-specific sub-packages are provided.

Two  classes  are involved  in  obtaining  data.   An acquisition  class
(subclass  of   :py:class:`~base.BaseAcquisition`)  provides  the  basic
interface to  the data source,  setting up any connections  required.  A
fetcher class (subclass of :py:class:`~base.BaseDataFetcher`) is used to
get data from a specified channel  and shot number.  In general usage, a
fetcher    class    is   not    handled    directly,    but   via    the
:py:meth:`~base.BaseAcquisition.getdata` method. For example::

 >>> import pyfusion   
 >>> h1 = pyfusion.getDevice('H1')   
 >>> mirnov_data = h1.acq.getdata(58133, 'H1_mirnov_array_1_coil_1')

Here,  ``h1`` is an  instance of  :py:class:`~devices.H1.device.H1` (the
subclass   of    :py:class:`~devices.base.Device`   specified   in   the
``[Device:H1]`` section  in the configuration  file). When instantiated,
the device class  checks the configuration file for  a acquisition class
specification,  and attaches  an instance  of the  specified acquisition
class, here ``h1.acq`` (which  is a synonym of ``h1.acquisition``).  The
:py:meth:`~base.BaseAcquisition.getdata`    method    checks    for    a
configuration    section     (here    it    is     a    section    named
``[Diagnostic:H1_mirnov_array_1_coil_1]``)  with  information about  the
diagnostic including which data fetcher  class to use.  The data fetcher
is then called to fetch and return the data."""

from pyfusion.acquisition.utils import get_acq_from_config
