

#################################
Installing pyfusion on Windows XP
#################################

:Release: |version|
:Date: |today|


This procedure uses native Windows installs, an alternative method is to use cygwin.

Requirements
------------

`Enthought Python Distribution <http://www.enthought.com/products/edudownload.php>`_ contains python, numpy, matplotlib and other libraries used by pyfusion

Recommended
-----------

`msysgit <http://code.google.com/p/msysgit/>`_ is a windows version of the Git distributed version control system, used to maintain pyfusion. 


Installing Pyfusion
-------------------

If you haven't already got a local directory in your PYTHONPATH, add one, e.g: make a directory ``C:\Documents and Settings\user\code\python`` and, using ``My Computer -> Properties -> Advanced -> Environment Variables`` set ``PYTHONPATH`` to ``%PYTHONPATH%;C:\Documents and Settings\user\code\python``.


If you have msysgit installed, use ``Start->All programs->Git->Git Bash``::

  cd code/python
  git clone git://github.com/dpretty/pyfusion.git
  cd pyfusion
  git checkout -b dev origin/dev



Making a custom configuration file
----------------------------------

Edit the file ``C:\Documents and Settings\user\.pyfusion\pyfusion.cfg``
