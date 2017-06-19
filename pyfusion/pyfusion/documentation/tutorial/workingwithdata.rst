.. _tut-workingwithdata:

*****************
Working with Data
*****************

Filters
^^^^^^^

Filters in pyfusion are methods which take a data object, modify the data, and return another (or the same) data object. Generally, filters are attached to data classes as methods::
 
 >>> data = h1.getData(.....)
 >>> data.reduce_time([0.02,0.03])


For example, the internal pyfusion code which defines the ``reduce_time()`` filter looks like this::

 @register("TimeseriesData", "DataSet")
 def reduce_time(input_data, new_time_range):
     from pyfusion.data.base import DataSet
     if isinstance(input_data, DataSet):
         output_dataset = input_data.copy()
         output_dataset.clear()
         for data in input_data:
             try:
                 output_dataset.add(data.reduce_time(new_time_range))
             except AttributeError:
                 pyfusion.logger.warning("Data filter 'reduce_time' not applied to item in dataset")
         return output_dataset 

     new_time_args = searchsorted(input_data.timebase, new_time_range)
     input_data.timebase =input_data.timebase[new_time_args[0]:new_time_args[1]]
     if input_data.signal.ndim == 1:
         input_data.signal = input_data.signal[new_time_args[0]:new_time_args[1]]
     else:
         input_data.signal = input_data.signal[:,new_time_args[0]:new_time_args[1]]
     return input_data



