""" Useful functions for manipulating config files."""

from ConfigParser import NoSectionError
import PyFusionGUI.pyfusion

def CannotImportFromConfigError(Exception):
    """Failed to import a module, class or method from config setting."""
    
def import_from_str(string_value):
    # TODO: make shortcuts for loading from within pyfusion
    split_val = string_value.split('.')
    val_module = __import__('.'.join(split_val[:-1]),
                            globals(), locals(),
                            [split_val[-1]])
    return val_module.__dict__[split_val[-1]]

def import_setting(component, component_name, setting):
    """Attempt to import and return a config setting."""
    value_str = pyfusion.config.pf_get(component, component_name, setting)
    return import_from_str(value_str)

def kwarg_config_handler(component_type, component_name, **kwargs):
    for config_var in pyfusion.config.pf_options(component_type, component_name):
            if not config_var in kwargs.keys():
                kwargs[config_var] = pyfusion.config.pf_get(component_type,
                                                   component_name, config_var)
    return kwargs


def get_config_as_dict(component_type, component_name):
    config_option_list = pyfusion.config.pf_options(component_type, component_name)
    config_map = lambda x: (x, pyfusion.config.pf_get(component_type, component_name, x))
    return dict(map(config_map, config_option_list))


def read_config(config_files):
    """Read config files.

    Argument is either a single file object, or a list of filenames.
    """
    try:
        existing_database = pyfusion.config.get('global', 'database')
    except NoSectionError:
        existing_database = 'None'

    try:
        files_read = pyfusion.config.readfp(config_files)
    except:
        files_read = pyfusion.config.read(config_files)

    if files_read != None: # readfp returns None
        if len(files_read) == 0: 
            raise LookupError, str('failed to read config files from [%s]' %
                                   (config_files))

    # print(config_files)
    # config_database  = pyfusion.config.get('global', 'database')
    #
    # if config_database.lower() != existing_database.lower():
    #     pyfusion.orm_manager.shutdown_orm()
    #     if config_database.lower() != 'none':
    #         pyfusion.orm_manager.load_orm()

def clear_config():
    """Clear pyfusion.config."""
    import pyfusion
    pyfusion.config = pyfusion.conf.PyfusionConfigParser()
