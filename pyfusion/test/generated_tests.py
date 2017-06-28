import pyfusion
from pyfusion.test.tests import find_subclasses, PfTestBase, SQLTestCase, NoSQLTestCase, TEST_FLAGS



for test_class in find_subclasses(pyfusion, PfTestBase):
    globals()['TestSQL%s' %test_class.__name__] = type('TestSQL%s' %test_class.__name__, (test_class, SQLTestCase), {})
    globals()['TestSQL%s' %test_class.__name__].sql = True
    globals()['TestSQL%s' %test_class.__name__].generated = True
    globals()['TestNoSQL%s' %test_class.__name__] = type('TestNoSQL%s' %test_class.__name__, (test_class, NoSQLTestCase), {})
    globals()['TestNoSQL%s' %test_class.__name__].sql = False
    globals()['TestNoSQL%s' %test_class.__name__].generated = True
    for flag in TEST_FLAGS:
        if hasattr(test_class, flag):
            setattr(globals()['TestSQL%s' %test_class.__name__], flag, getattr(test_class, flag))
            setattr(globals()['TestNoSQL%s' %test_class.__name__], flag, getattr(test_class, flag))

