"""Utilities for malipulating acquisition objects."""

from pyfusion import config
from pyfusion.conf.utils import import_setting

def get_acq_from_config(acq_name):
    """Return  the acquisition  class  specified by  `acq_class` in  the
    `[Acquisition:acq_name]` section of the configuration file.

    :param acq_name: name of the acquisition configuration section.
    :returns: acquisition class specified by `acq_class` in the \
    `[Acquisition:acq_name]` section of the configuration file
    """
    acq_class = import_setting('Acquisition', acq_name, "acq_class")
    return acq_class

def getAcquisition(acq_name):
    """Return  the an  instance of  the acquisition  class  specified by
    `acq_class`   in  the   `[Acquisition:acq_name]`   section  of   the
    configuration file.

    :param acq_name: name of the acquisition configuration section.
    :returns: instance of the acquisition class specified by \
    `acq_class` in the `[Acquisition:acq_name]` section of the \
    configuration file
    """
    acq_class = get_acq_from_config(acq_name)
    return acq_class(acq_name)
