Object relational mapping
=========================

Overview
--------

`Object relational mapping <http://en.wikipedia.org/wiki/Object-relational_mapping>`_ (ORM) is a method of maintaining a connection between a `relational database <http://en.wikipedia.org/wiki/Relational_database>`_ (e.g. `MySQL <http://en.wikipedia.org/wiki/MySQL>`_) and `object orientated <http://en.wikipedia.org/wiki/Object_oriented>`_ programming languages (e.g. `python <http://en.wikipedia.org/wiki/Python_(programming_language)>`_). The types of data used with pyfusion are very well suited to being stored in relational databases, i.e. we deal with a large number of items which all share the same set of attributes. With an ORM, we get the benefits of both fast and efficient `SQL <http://en.wikipedia.org/wiki/SQL>`_ querying of the data, and object orientated code. 

In pyfusion, the ORM is activated by setting the ``database`` configuration variable in the ``[global]`` section of your configuration file. Pyfusion will not use ORM if  ``database`` is set to ``None``. 


SQLAlchemy
^^^^^^^^^^

`SQLAlchemy <http://www.sqlalchemy.org>`_ is a python library which provides a comprehensive interface to relational databases, and includes ORM. The documentation is available here: http://www.sqlalchemy.org/docs/ 

The pyfusion ORM configuration
------------------------------

Pyfusion uses SQLAlchemy for its ORM. The standard method for configuring an ORM with SQLAlchemy is to explicitly construct Table objects and link them to Python classes with a mapping object. An alternative configuration is to use the SQLAlchemy declarative extension, which provides a base class which provides Table and mapper attributes to any class which inherits it. The two approaches represent different styles code rather than providing different functionality. Pyfusion uses the standard approach to keep ORM code separate from non-ORM (class definitions) code, allowing pyfusion to be used without ORM.

..

Module-wide configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

Code which controls module-wide (i.e. all of pyfusion) ORM is located in ``pyfusion.orm``. The important components here are the ORM engine, session and metadata. These are created by ``pyfusion.orm.setup_orm()`` which is called after the configuration files are read during ``import pyfusion``.


Engine
""""""

The SQLAlchemy engine provides an abstraction of the relational database (beneath it could be MySQL, Postgres, SQLite, etc), and a pool of connections to the database. Starting a database connection is an expensive operation, to streamline database interaction, the engine keeps a pool of connections which it uses and recycles to avoid the overhead of creating database connections for each operation.::

    pyfusion.orm_engine = create_engine(pyfusion.config.get('global', 'database'))


Session
"""""""

An instance of the  SQLAlchemy ``Session`` class is used to manage interactions with the database, it can keep track of modifications to data instances and flush multiple changes to the database when required. We use ``scoped_session`` to provide a thread-local ``Session`` instance, which allows us to use the same session in different parts of pyfusion. The session configuration looks like::

 pyfusion.Session = scoped_session(sessionmaker(autocommit=False,
                                   autoflush=True,
                                   bind=pyfusion.orm_engine))


The ``autocommit`` and ``autoflush`` arguments  prescribe how the session should organise transactions. A `database transaction <http://en.wikipedia.org/wiki/Database_transaction>`_ refers to a group of queries which should be treated as a single operation on the database, either all queries in a should be applied, or none of them should. Using ``commit()`` in an sqlalchemy session commits the current transaction, whereas ``flush()`` will write pending data to the database without closing the transaction. In ``autocommit`` mode SQLAlchemy automatically commits after each ``flush()``, while this removes some flexibility in construction of transactions it can be useful for testing and debug purposes. Regardless of these settings, ``commit()`` will always call a ``flush()`` before committing the transaction. The ``autoflush=True`` argument specifies that ``flush()`` should be called before any individual query is issued.  


Class-level configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The ORM code for classes in pyfusion follows the class code, and is active only if ``pyfusion.USE_ORM`` is ``True``. The ORM class code contains a ``Table`` definition, a call to ``pyfusion.metadata.create_all()`` to create table, and a mapping of the class to the table. For example, the ``Device`` class definition and ORM code appears as::

 class Device(object):
 
     def __init__(self, config_name, **kwargs):
         if pyfusion.config.pf_has_section('Device', config_name):
             self.__dict__.update(get_config_as_dict('Device', config_name))
         self.__dict__.update(kwargs)
         self.name = config_name
 
         #### attach acquisition
         if hasattr(self, 'acq_name'):
             acq_class_str = pyfusion.config.pf_get('Acquisition',
                                           self.acq_name, 'acq_class')
             self.acquisition = import_from_str(acq_class_str)(self.acq_name)
             # shortcut
             self.acq = self.acquisition
         else:
             pyfusion.logging.warning(
                 "No acquisition class specified for device")
 
 
 if pyfusion.USE_ORM:
     from sqlalchemy import Table, Column, Integer, String
     from sqlalchemy.orm import mapper
     device_table = Table('devices', pyfusion.metadata,
                          Column('id', Integer, primary_key=True),
                          Column('name', String, unique=True))
 
     pyfusion.metadata.create_all()
     mapper(Device, device_table)



Does pyfusion read from the config file or data database?
---------------------------------------------------------

notes:
e.g. when a device is created which has a config definition, it will be loaded from sql if it exists. if it doesnt exist it will be created. at present there is no checking to make sure that the sql version matches the params of the config. there is no automated way of changing the sql version if you change the config - this shoulnt be done anyway, as other data may have been created with the existing device, diagnostic etc and it we dont want to have processed data attached to an instance which is not responsible for its creation... etc...


