"""..."""

import pyfusion

def orm_register():
    def reg_item(orm_func):
        pyfusion.orm_manager.add_reg_func(orm_func)
        return orm_func
    return reg_item

