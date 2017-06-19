Internals
=========

Static internal values
----------------------

The following are set when the pyfusion module is imported, and are not 
designed to be changed by the user.


PYFUSION_ROOT_DIR
^^^^^^^^^^^^^^^^^

A string containing the directory where the pyfusion module resides.

LOGGING_CONFIG_FILE
^^^^^^^^^^^^^^^^^^^

 .. TODO: should pyfusion.logger be :mod: ??
 .. TODO: link pyfusion.logger to its own doc page.

Filename of the configuration file used to set up :mod:`pyfusion.logger`.


VERSION
^^^^^^^

Pyfusion version, as returned by :func:`pyfusion.version.get_version`.


DEFAULT_CONFIG_FILE
^^^^^^^^^^^^^^^^^^^

Location of file containing default pyfusion configuration. 

USER_PYFUSION_DIR
^^^^^^^^^^^^^^^^^

Location of user's pyfusion directory. On linux this will be ``$HOME/.pyfusion``

USER_CONFIG_FILE
^^^^^^^^^^^^^^^^

Location of user's pyfusion configuration file. On linux this will be
``$HOME/.pyfusion/pyfusion.cfg``


USER_ENV_CONFIG_FILE
^^^^^^^^^^^^^^^^^^^^

This is set to the value of the ``PYFUSION_CONFIG_FILE`` environment variable,
which supersedes ``USER_CONFIG_FILE`` and ``DEFAULT_CONFIG_FILE``.


