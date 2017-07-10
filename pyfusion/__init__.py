# import os, logging
# import logging.config
#
# from pyfusion.conf import PyfusionConfigParser
# from pyfusion.conf.utils import read_config
# from pyfusion.orm import ORMManager
# from pyfusion.version import get_version
#
# # This grabs the directory of the pyfusion module.
# PYFUSION_ROOT_DIR = os.path.dirname(__file__)
#
# # Location of the logging configuration file.
# LOGGING_CONFIG_FILE = os.path.join(PYFUSION_ROOT_DIR, 'conf', 'logging.cfg')
#
# # Grab the pyfusion version
# VERSION = get_version()
#
# # This creates a parser to process and store configuration file data.
# # PyfusionConfigParser is a subclass of ConfigParser.ConfigParser in the
# # python standard library. It is customised to parse [Type:Name] sections
# # in configuration files.
# config = PyfusionConfigParser()
#
# # This allows us to activate and de-activate the object relational
# # mapping (ORM) code from within a python session, and also helps to
# # keep the ORM code separate so it doesn't slow things down for
# # users who don't want the database backend.
# orm_manager = ORMManager()
#
# # This sets up an instance of logger from the python standard library.
# logging.config.fileConfig(LOGGING_CONFIG_FILE)
# logger = logging.getLogger("pyfusion")
#
# # Find the default pyfusion configuration file...
# DEFAULT_CONFIG_FILE = os.path.join(PYFUSION_ROOT_DIR, 'pyfusion.cfg')
#
# # ... and the user's custom configuration file. First, if they don't
# # already have a folder for pyfusion stuff then let's make one
# USER_PYFUSION_DIR = os.path.join(os.path.expanduser('~'), '.pyfusion')
# if not os.path.exists(USER_PYFUSION_DIR):
#     os.mkdir(USER_PYFUSION_DIR)
# # and here is the custom user configuration file
# USER_CONFIG_FILE = os.path.join(USER_PYFUSION_DIR, 'pyfusion.cfg')
#
# # Also allow specification of other configuration files from
# # a PYFUSION_CONFIG_FILE environment variable
# USER_ENV_CONFIG_FILE = os.getenv('PYFUSION_CONFIG_FILE','')
#
# # Now we actually load the configuration files. Settings in
# # DEFAULT_CONFIG_FILE will be superseded by those in USER_CONFIG_FILE,
# # and USER_ENV_CONFIG_FILE will supersede both. As well as storing the
# # settings, read_config() will set up the ORM backend if required.
# read_config([DEFAULT_CONFIG_FILE, USER_CONFIG_FILE, USER_ENV_CONFIG_FILE])
#
# # We import these into the base pyfusion namespace for convenience.
# from pyfusion.devices.base import getDevice
# from pyfusion.acquisition.utils import getAcquisition
#
