"""Object relational mapping for Pyfusion"""

import pyfusion

class ORMManager(object):
    def __init__(self):
        self.func_list = []
        self.IS_ACTIVE = False
    def add_reg_func(self, orm_func):
        self.func_list.append(orm_func)
        if self.IS_ACTIVE:
            orm_func(self)
    def setup_session(self):
        db_string = pyfusion.config.get('global', 'database')

        if db_string.lower() != 'none':
            from sqlalchemy import create_engine, MetaData
            from sqlalchemy.orm import scoped_session, sessionmaker, clear_mappers
        
            self.engine = create_engine(pyfusion.config.get('global', 'database'))
            self.Session = scoped_session(sessionmaker(autocommit=False,
                                                       autoflush=True,
                                                       bind=self.engine,
                                                       expire_on_commit=False))
        
            self.metadata = MetaData()
            self.metadata.bind = self.engine
            self.metadata.create_all()
        
            self.IS_ACTIVE = True

            # for use in tests
            self.clear_mappers = clear_mappers


    def load_orm(self):
        self.setup_session()
        for f in self.func_list:
            f(self)
        self.metadata.create_all()

    def shutdown_orm(self):
        # TODO: should we be using clear_mappers here?
        # probably not, although this doesn't get called often
        # Also, this might not be the best way to shut down sqlalchemy.
        if self.IS_ACTIVE:
            self.clear_mappers()
            self.Session.close_all()
            del self.Session
            del self.metadata
            del self.engine

        self.IS_ACTIVE = False
    
