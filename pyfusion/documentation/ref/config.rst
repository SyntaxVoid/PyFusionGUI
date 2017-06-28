.. _config-files:

Configuration files
"""""""""""""""""""

Overview
--------

Pyfusion uses simple text files to store information such as data acquisition settings, diagnostic coordinates, SQL database configurations, etc. A pyfusion configuration file looks something like this::

 [global]
 database = sqlite:///:memory:

 [Device:H1]
 dev_class = pyfusion.devices.H1.device.H1
 acq_name = MDS_h1
 
 [Acquisition:MDS_h1]
 acq_class = pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition
 server = h1data.anu.edu.au
 
 [CoordTransform:H1_mirnov]
 magnetic = pyfusion.devices.H1.coords.MirnovKhMagneticCoordTransform
 
 [Diagnostic:H1_mirnov_array_1_coil_1]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_1
 coords_cylindrical = 1.114, 0.7732, 0.355
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1_coil_2]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_2
 coords_cylindrical = 1.185, 0.7732, 0.289
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1_coil_3]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_3
 coords_cylindrical = 1.216, 0.7732, 0.227
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1]
 data_fetcher = pyfusion.acquisition.base.MultiChannelFetcher
 channel_1 = H1_mirnov_array_1_coil_1
 channel_2 = H1_mirnov_array_1_coil_2
 channel_3 = H1_mirnov_array_1_coil_3




There are two types of sections in this file: there is one `special` section (global) and several `component` sections (e.g. Device:H1, Acquisition:MDS_h1, CoordTransform:H1_mirnov, etc.)

  .. ********** EDIT LINE.  ***********



The sections in the configuration (except for [variabletypes]) file have the syntax
[Component:name], where Component is one of: Acquisition, Device,
Diagnostic. When instantiating a class, such as Device, Acquisition,
Diagnostic, etc. which looks in the configuration file for settings,
individual settings can be overridden using the corresponding keyword
arguments. For example, ``Device('my_device')`` will use settings in
the ``[Device:my_device]`` configuration section, and
``Device('my_device', database='sqlite://')`` will override the
database configuration setting with ``sqlite://`` (a temporary in-memory database).  


The pyfusion configuration parser :class:`pyfusion.conf.PyfusionConfigParser` is a simple subclass of the `standard
python configparser
<http://docs.python.org/library/configparser.html>`_, for example, to
see the configuration sections, type::

    pyfusion.config.sections()



Loading config files
--------------------
When pyfusion is imported, will load the default configuration file
provided in the source code (that is in the pyfusion directory)
followed by your custom configuration file, 
in ``$HOME/.pyfusion/pyfusion.cfg``, if it exists. 
and finally files pointed to by the environment variable PYFUSION_CONFIG_FILE
if they exist. This allows temporarily overriding config variables.

Additional config files can be loaded with ``pyfusion.read_config()``::

	   pyfusion.read_config(["another_config_filename_1", "another_config_filename_2"])

The ``read_config`` argument can either be a single file-like object
(any object which has a ``readlines()`` method) or a list of
filenames, as shown above. If you do not supply any argument,
``read_config()`` will load the default configuration files (the same
ones loaded when you import pyfusion). 

To clear the loaded pyfusion configuration, use
``pyfusion.clear_config()``. If you want to return the configuration
to the default settings (the configuration you have when you import
pyfusion), type::

	   pyfusion.clear_config()
	   pyfusion.read_config()




[variabletypes]
---------------
variabletypes is a section for defining the types (integer, float,
boolean) of variables specified throughout the configuration file. By
default, variables are assumed to be strings (text) - only variables
of type integer, float or boolean should be listed here.

For example, if three variables (arguments) for the Diagnostic class
are n_samples (integer), sample_freq (float) and normalise (boolean)
the syntax is:: 

	Diagnostic__n_samples = int
	Diagnostic__sample_freq = float
	Diagnostic__normalise = bool

Note the double underscore (__) separating the class type and the
variable name.

[Device:name]
-------------

database
~~~~~~~~

Location of database in the `SQLAlchemy database URL syntax`_. 

e.g.::
   
   no example yet

.. _SQLAlchemy database URL syntax: http://www.sqlalchemy.org/docs/04/dbengine.html#dbengine_establishing

acq_name
~~~~~~~~

Name of Acquisition config setting ( [Acquisition:acq_name] ) to be used for this device.

e.g.::

   acq_name = test_fakedata

dev_class
~~~~~~~~~

Name of device class (subclass of pyfusion.devices.base.Device)
to be used for this device. This is called when using the convenience
function pyfusion.getDevice. For example, if the configuration file
contains::

	[Device:my_tjii_device]
	dev_class = pyfusion.devices.TJII

then using::

     import pyfusion
     my_dev = pyfusion.getDevice('my_tjii_device')

``my_dev`` will be an instance of pyfusion.devices.TJII

[Acquisition:name]
------------------

acq_class
~~~~~~~~~

Location of acquisition class (subclass of pyfusion.acquisition.base.BaseAcquisition). 

e.g.::
  
   acq_class = pyfusion.acquisition.fakedata.FakeDataAcquisition

[Diagnostic:name]
-----------------


data_fetcher
~~~~~~~~~~~~

Location of class (subclass of pyfusion.acquisition.base.BaseDataFetcher) to fetch
the data for the diagnostic.

tests.cfg
---------

A seperate configuration file "tests.cfg", in the same ".pyfusion" folder in your home directory, can be used during development to enable tests which are disabled by default.

An example of the syntax is::

	[EnabledTests]
	mdsplus = True
	flucstrucs = True

etc...




Database
--------
The database layer is handled by `SQLAlchemy <http://www.sqlalchemy.org>`_ 

.. _db-urls:

Database URL
~~~~~~~~~~~~

Database URLs are the same as for SQLAlchemy::

	 driver://username:password@host:port/database

Simplifying changes by substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The syntax %(sym)s will substitute the contents of sym.  e.g.
fetchr =  pyfusion.acquisition.H1.fetch.H1LocalTimeseriesDataFetcherh1datafetcher
data_fetcher = %(fetchr)s

This way only one edit needs to be made to change all diagnostics, if
the definition is fetchr is in the special [DEFAULT] section

User Defined Sections
~~~~~~~~~~~~~~~~~~~~~
We will probably include a section Plots containing things like
FT_Axis = [0, 0.08, 0, 300000]
to provide defaults for the Frequency-Time axis etc
Note that such settings are highly device dependent and although they
will be recognised in the code, they usually should not be given
values in code distributions.

The User could put their own items in there or other sections to avoid 

For more details, refer to http://www.sqlalchemy.org/docs/05/dbengine.html#create-engine-url-arguments 

