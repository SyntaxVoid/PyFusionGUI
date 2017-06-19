from pyfusion.test.tests import PfTestBase
import pyfusion

class CheckORM(PfTestBase):
    def test_orm(self):
        from pyfusion import orm

class CheckORMManager(PfTestBase):
    def test_orm_manager(self):
        from sqlalchemy import create_engine, MetaData
        from sqlalchemy.orm import scoped_session, sessionmaker
        from sqlalchemy.engine.base import Engine
        from sqlalchemy.orm.scoping import ScopedSession
        from pyfusion.orm import ORMManager

        self.assertIsInstance(pyfusion.orm_manager, ORMManager)

        ## for testing, close sessoins and clear mappers before calling load_orm()
        if pyfusion.orm_manager.IS_ACTIVE:
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()

        # check metadata, engine binding, etc
        pyfusion.orm_manager.setup_session()
        if pyfusion.orm_manager.IS_ACTIVE:
            self.assertIsInstance(pyfusion.orm_manager.engine, Engine)
            self.assertIsInstance(pyfusion.orm_manager.Session, ScopedSession)
            self.assertIsInstance(pyfusion.orm_manager.metadata, MetaData)
        
    def test_manager_reg(self):
        from pyfusion.orm.utils import orm_register
        from sqlalchemy import MetaData

        @orm_register()
        def test_orm_load(man):
            man.a = 1
            man.b = 2

        self.assertTrue(test_orm_load in pyfusion.orm_manager.func_list)

        ## for testing, close sessoins and clear mappers before calling load_orm()
        if pyfusion.orm_manager.IS_ACTIVE:
            pyfusion.orm_manager.Session.close_all()
            pyfusion.orm_manager.clear_mappers()
        
            pyfusion.orm_manager.load_orm()

            self.assertEqual(pyfusion.orm_manager.a, 1)
            self.assertEqual(pyfusion.orm_manager.b, 2)
            # make sure the session, metadata, etc is started when we load_orm
            self.assertIsInstance(pyfusion.orm_manager.metadata, MetaData)

