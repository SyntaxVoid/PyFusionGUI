from pyfusion.devices.base import Device
from pyfusion.orm.utils import orm_register

class H1(Device):
    pass

@orm_register()
def orm_load_h1device(man):
    from sqlalchemy import Table, Column, Integer, ForeignKey
    from sqlalchemy.orm import mapper
    man.h1device_table = Table('h1device', man.metadata, 
                            Column('basedevice_id', Integer, ForeignKey('devices.id'), primary_key=True))
    mapper(H1, man.h1device_table, inherits=Device, polymorphic_identity='h1')
