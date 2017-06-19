.. _tut-getting:

************
Getting Data
************

Using the Device class
^^^^^^^^^^^^^^^^^^^^^^

The recommended method of retrieving data with pyfusion is by creating
an instance of :class:`Device` class (to represent LHD, H-1, TJ-II,
etc) and using the attached :meth:`getdata` method::

 >>> import pyfusion
 >>> h1 = pyfusion.getDevice('H1')
 >>> mirnov_data = h1.acq.getdata(58133, 'H1_mirnov_array_1_coil_1')


The :meth:`getDevice` method takes a single argument which corresponds to a Device entry in the pyfusion configuration::

   [Device:H1]
   dev_class = pyfusion.devices.H1.device.H1
   acq_name = MDS_h1

:meth:`getDevice` then returns an instance the specified subclass of  :class:`Device` (here, :class:`pyfusion.devices.H1.device.H1` is a subclass of :class:`Device`) initiated with the same argument, i.e. the following are synonyms::

 >>> h1 = pyfusion.getDevice('H1')


and::

 >>> from pyfusion.devices.H1.device import H1
 >>> h1 = H1('H1')  



Pre-loading of data acquisition system during Device instantiation.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data acquisition in pyfusion is handled by two components, an ``Acquisition`` class and a ``DataFetcher`` class. The Acquisition class sets up a connection with a data acquisition system; a  DataFetcher class uses this connection to fetch a requested data. When a new instance of a Device (sub)class is created, pyfusion will look to the configuration file to see if any data acquisition system is specified. In the example above, the ``H1`` configuration contains::

    acq_name = MDS_h1


which tells pyfusion to look for the data acquisition configuration::

  [Acquisition:MDS_h1]
  acq_class = pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition
  server = h1data.anu.edu.au

and attach a new instance of the specified ``acq_class`` to the device::
 
 >>> import pyfusion
 >>> h1 = pyfusion.getDevice('H1')
 >>> print h1.acquisition
 <pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition object at 0x96d460c>
 >>> h1.acq == h1.acquisition 
 True

Where ``h1.acq`` is simply a shortcut to ``h1.acquisition``. The connection to the data acquisition system is created when the ``Acquisition`` class is instantiated, in this example: ``mdsconnect()`` is called when ``h1.acquisition`` is created. 


Data acquisition via getdata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In our original example::
 
  >>> mirnov_data = h1.acq.getdata(58133, 'H1_mirnov_array_1_coil_1')

the ``getdata()`` method is essentially a wrapper around the DataFetcher class which uses the configuration setting to determine which subclass of DataFetcher should be used for the specified diagnostic. Here, ``h1.acq.getdata(58133, 'H1_mirnov_array_1_coil_1')`` looks up the configuration section ``[Diagnostic:H1_mirnov_array_1_coil_1]``::

 [Diagnostic:H1_mirnov_array_1_coil_1]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_1
 coords_cylindrical = 1.114, 0.7732, 0.355
 coord_transform = H1_mirnov

An instance of the class specified by ``data_fetcher`` (a subclass of ``DataFetcher``) is created with the parameters specified in the configuration. ``DataFetcher`` classes have a ``fetch()`` method, which returns the data as a pyfusion ``Data`` object; ``getdata()`` calls this ``fetch()`` method and returns the data object.  
