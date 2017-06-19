:mod:`acquisition` -- data acquisition 
======================================


.. automodule:: pyfusion.acquisition

Base classes
------------

.. autoclass:: pyfusion.acquisition.base.BaseAcquisition
   :members:
.. autoclass:: pyfusion.acquisition.base.BaseDataFetcher
   :members:
.. autoclass:: pyfusion.acquisition.base.MultiChannelFetcher
   :members:

Sub-packages for specific data sources
--------------------------------------

Custom       subclasses       :class:`~base.BaseAcquisition`       and
:class:`~base.BaseDataFetcher`  classes  are  contained  in  dedicated
sub-packages. Each sub-package has the structure::

 subpkg/
       __init__.py
       acq.py
       fetch.py

with       :mod:`acq.py`      containing      a       subclass      of
:class:`~base.BaseAcquisition`   and   :mod:`fetch.py`  containing   a
subclass of :class:`~base.BaseDataFetcher`.

:mod:`MDSPlus`
^^^^^^^^^^^^^^

.. automodule:: pyfusion.acquisition.MDSPlus

Classes
"""""""
.. autoclass:: pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition

.. autoclass:: pyfusion.acquisition.MDSPlus.fetch.MDSPlusDataFetcher

:mod:`H1`
^^^^^^^^^

.. automodule:: pyfusion.acquisition.H1

Classes
"""""""
.. autoclass:: pyfusion.acquisition.H1.fetch.H1DataFetcher


:mod:`LHD`
^^^^^^^^^^

.. automodule:: pyfusion.acquisition.LHD

:mod:`DSV`
^^^^^^^^^^

.. automodule:: pyfusion.acquisition.DSV

Classes
"""""""
.. autoclass:: pyfusion.acquisition.DSV.acq.DSVAcquisition

.. autoclass:: pyfusion.acquisition.DSV.fetch.DSVMultiChannelTimeseriesFetcher

:mod:`FakeData`
^^^^^^^^^^^^^^^

.. automodule:: pyfusion.acquisition.FakeData

Classes
"""""""
.. autoclass:: pyfusion.acquisition.FakeData.acq.FakeDataAcquisition

.. autoclass:: pyfusion.acquisition.FakeData.fetch.SingleChannelSineFetcher





