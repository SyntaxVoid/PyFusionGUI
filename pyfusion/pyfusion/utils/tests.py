"""Test code for test code."""

from pyfusion.test.tests import PfTestBase


class CheckPing(PfTestBase):
    """Test the ping utility"""

    def testPing(self):
        from pyfusion.utils.net import ping


class Dummy:
    def __init__(self, **kwargs):
        for kwa in kwargs.keys():
            self.__dict__[kwa] = kwargs[kwa]


class CheckEqualExceptFor(PfTestBase):
    """Test custom object comparison, which allows specified
    attributed to be ignored"""

    def test_equal_except_for(self):
        from pyfusion.utils.debug import equal_except_for
        d1 = Dummy(a=1, b=2, abc=3)
        d2 = Dummy(a=1, b=2)
        self.assertFalse(equal_except_for(d1, d2))
        self.assertTrue(equal_except_for(d1, d1))
        self.assertTrue(equal_except_for(d1, d2, ['b', 'abc']))
        self.assertTrue(equal_except_for(d1, d2, 'abc'))

