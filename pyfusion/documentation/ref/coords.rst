.. _coords:

Coordinates
"""""""""""

Coordinates can be defined in the configuration file, for example pyfusion.cfg contains::

   [CoordTransform:H1_mirnov]
   magnetic = pyfusion.devices.H1.coords.MirnovKhMagneticCoordTransform

   [Diagnostic:H1_mirnov_array_1_coil_1]
   data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
   mds_path = \h1data::top.operations.mirnov:a14_14:input_1
   coords_cylindrical = 1.114, 0.7732, 0.355
   coord_transform = H1_mirnov


which specifies coords_cylindrical as the canonical coordinate representation, followed by a reference to a magnetic coordinate transform which depends on magnetic configuration (and therefore varies from shot to shot).

When data is acquired, eg::

   import pyfusion as pf
   h1 = pf.getDevice('H1')
   data = h1.acq.getdata(58010, 'H1_mirnov_array_1_coil_1')

