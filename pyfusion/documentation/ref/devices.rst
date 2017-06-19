:mod:`devices` -- Representation of fusion device
=================================================
.. module:: devices
   :synopsis: Representation of fusion device

In general usage, a customised subclass will be used rather than direct instantiation of classes in this module. The :func:`getDevice` helper function is provided to simplify the use of device classes. If we have in our configuration file::
   
   [Device:MyDevice]
   dev_class = pyfusion.devices.H1.device.H1
   param_1 = param_1_value

then :func:`getDevice` will return an instance of dev_class::

   >>> import pyfusion
   >>> my_device = pyfusion.getDevice("MyDevice")
   >>> my_device
   <pyfusion.devices.H1.device.H1 instance at 0xa1d26ec>
   >>> my_device.param_1
   'param_1_value'

   

Base Device Objects
-------------------

.. module:: devices.base

The following classes are provided by the :mod:`~pyfusion.devices.base` submodule. In general, a customised subclass of :class:`Device` is used rather than the base class itself. 

.. class:: Device(config_name=None, **kwargs)

   Returns an instance of :class:`Device` optionally initialised with an underlying instance of a specified subclass of :class:`~pyfusion.acquisition.base.BaseAcquisition`.

   :arg config_name: configuration file section (i.e. [Device:config_name]) to load.
   
   Any configuration file option can be overridden by supplying an argument of the same name to :class:`Device`. For example, given a configuration file::
      
      [Device:MyDevice]
      param_1 = param_1_value
      param_2 = param_2_value

   the configuration will be loaded by specifying the config_name::   

      >>> import pyfusion
      >>> my_device = pyfusion.devices.base.Device(config_name="MyDevice")
      >>> my_device.param_1
      'param_1_value'
      >>> my_device.param_2
      'param_2_value'
      >>> my_other_device = pyfusion.devices.base.Device(config_name="MyDevice", param_1="some_other_value")
      >>> my_other_device.param_1
      'some_other_value'

   The only configuration parameter which is directly handled by :class:`Device` is "acq_name", which specifies the name of the data acquisition system (i.e. [Acquisition:MyAcquisition] in the configuration file) to be attached to the :class:`Device` instance. So, for a configuration file::
   
      [Device:MyDevice]
      acq_name = MyAcquisition
      param_1 = param_1_value
      param_2 = param_2_value

      [Acquisition:MyAcquisition]
      acq_class = pyfusion.acquisition.base.BaseAcquisition

   we get::

      >>> import pyfusion
      >>> my_device = pyfusion.devices.base.Device(config_name="MyDevice")
      >>> my_device.acquisition    
      <pyfusion.acquisition.base.BaseAcquisition object at 0x87a3a4c>
      >>> my_device.acq
      <pyfusion.acquisition.base.BaseAcquisition object at 0x87a3a4c>

   where my_device.acquisition (and synonym my_device.acq) is an instance of :class:`~pyfusion.acquisition.base.BaseAcquisition`.


H-1 Device Class
-------------------

.. module:: devices.H1.device

.. class:: H1

   Trivial subclass of :class:`Device` which doesn't add anything new (yet).
