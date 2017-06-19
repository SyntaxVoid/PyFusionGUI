"""LHD acquisition."""


from pyfusion.acquisition.base import BaseAcquisition

class LHDAcquisition(BaseAcquisition):
    pass
"""
    def __init__(self, *args, **kwargs):
        from MDSplus import Data
        self._Data = Data
        super(MDSPlusAcquisition, self).__init__(*args, **kwargs)
        self._Data.execute("mdsconnect('%(server)s')" %{'server':self.server})

    def __del__(self):
        self._Data.execute("mdsdisconnect()")
"""
