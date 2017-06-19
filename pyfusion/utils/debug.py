"""Utilities useful in debugging"""

def equal_except_for(a, b, except_arg=None):
    """Compare objects via __dict__. allow for ignored attributes.

    (Ugly code - but it works)
    """
    a_dict = a.__dict__.copy()
    b_dict = b.__dict__.copy()
    if except_arg != None:
        if hasattr(except_arg, '__iter__'):
            for arg in except_arg:
                for dict_i in [a_dict, b_dict]:
                    if dict_i.has_key(arg):
                        dict_i.pop(arg)
        else:
            for dict_i in [a_dict, b_dict]:
                if dict_i.has_key(except_arg):
                    dict_i.pop(except_arg)
            
    return a_dict == b_dict
