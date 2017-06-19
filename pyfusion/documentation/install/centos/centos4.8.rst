.. _install-centos4.8:

#################################
Installing pyfusion on CentOS 4.8
#################################

:Release: |version|
:Date: |today|


Because CentOS 4.8 has an old version of python (version 2.3), which won't work with some of the libraries required by pyfusion, we will install a newer version of python and some libraries in the user's home directory.



---------------------
Environment variables
---------------------

First, set up some environment variables::

   export PYTHONPATH=$PYTHONPATH:$HOME/code/python
   mkdir -p code/python

   export SOURCEDIR=$HOME/source
   mkdir $SOURCEDIR
   export LOCALDIR=$HOME/local
   mkdir $LOCALDIR

   export PATH=$LOCALDIR/bin:$PATH

The PYTHONPATH and PATH (and LOCALDIR if PATH refers to it) exports should also go in your ``$HOME/.bashrc`` file. 

------
Python
------

Now install Python - the version number is the latest in December 2010, you can use a later one if it exists, but don't use Python 3.x::

  cd $SOURCEDIR
  wget http://www.python.org/ftp/python/2.7.1/Python-2.7.1.tar.bz2
  tar -xjf Python-2.7.1.tar.bz2
  cd Python-2.7.1
  ./configure --prefix=$LOCALDIR
  make
  make install


Make sure the default python is now python2.7 (it should be this because $LOCALDIR/bin is first in your PATH environment variable)::

  > python

  Python 2.7.1 (r271:86832, Dec 26 2010, 03:33:20)
  [GCC 3.4.6 20060404 (Red Hat 3.4.6-11)] on linux2
  Type "help", "copyright", "credits" or "license" for more information.
  >>>



----------
Setuptools
----------

We need setuptools so we can install pip::

  cd $SOURCEDIR
  wget http://pypi.python.org/packages/2.7/s/setuptools/setuptools-0.6c11-py2.7.egg
  sh setuptools-0.6c11-py2.7.egg --prefix=$LOCALDIR


---
pip
---

We use pip to install other libraries::

  cd $SOURCEDIR
  wget http://pypi.python.org/packages/source/p/pip/pip-0.8.2.tar.gz
  tar -xzf pip-0.8.2.tar.gz
  cd pip-0.8.2
  python setup.py install --prefix=$LOCALDIR


-----
numpy
-----

Either using sudo or log in as root to install the dependencies::

  yum install blas lapack


Then use pip to install numpy::

  pip -v install numpy


----------
sqlalchemy
----------

Install using pip::

  pip -v install sqlalchemy

-------
ipython
-------

Optional, but very useful::

  pip -v install ipython

---
git
---

Git is used for pyfusion revision control, and makes it easy for you to update pyfusion and help with development::


  cd $SOURCEDIR
  wget http://kernel.org/pub/software/scm/git/git-1.7.3.4.tar.bz2
  tar -xjf git-1.7.3.4.tar.bz2
  cd git-1.7.3.4
  ./configure --prefix=$LOCALDIR
  make
  make install

--------
pyfusion
--------

We install with git::

  cd $HOME/code/python
  git clone git://github.com/dpretty/pyfusion.git
  cd pyfusion
  git checkout -b dev origin/dev


----------------
Setting up mysql
----------------

Because pyfusion uses sqlalchemy, you can choose from many different types of SQL servers, here we show how to set up MySQL.

As root (or sudo), install the required packages::

  yum install mysql mysql-server mysql-devel

Still as root, start MySQL::

  /etc/init.d/mysql start

and create a MySQL user for pyfusion::

  mysql
  > GRANT ALL PRIVILEGES ON *.* TO 'pyfusionuser'@'localhost' IDENTIFIED BY 'mypassword' WITH GRANT OPTION;


Now install (not as root) the python MySQL libraries::

  pip -v install MySQL-python

and create a test database to use with pyfusion::

  mysql -p
  > create database pyfusion_test;



Now, edit $HOME/.pyfusion/pyfusion.cfg to tell pyfusion to use this database, if the directory doesn't exist, make it::

  mkdir $HOME/.pyfusion

and then in ``$HOME/.pyfusion/pyfusion.cfg``::

  [global]
  database = mysql://pyfusionuser:mypassword@localhost/pyfusion_test


----------
Matplotlib
----------

You'll also want matplotlib installed to visualise the data. As root, install the dependencies (we'll use pyqt for the graphics backend)::

  yum install freetype-devel libpng-devel qt-devel

And as user::

  cd $SOURCEDIR
  # for some reason pip -v install matplotlib failed for me, so I downloaded the source file separately:
  wget http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-1.0/matplotlib-1.0.0.tar.gz/download
  pip -v install matplotlib-1.0.0



Now we setup the pyqt backend::

  cd $SOURCEDIR
  wget http://www.riverbankcomputing.com/static/Downloads/sip4/sip-4.12.tar.gz
  tar -xzf sip-4.12.tar.gz
  cd sip-4.12
  python configure.py
  make
  make install

  cd $SOURCEDIR
  wget http://www.riverbankcomputing.co.uk/static/Downloads/PyQt3/PyQt-x11-gpl-3.18.1.tar.gz
  tar -xzf PyQt-x11-gpl-3.18.1.tar.gz
  cd PyQt-x11-gpl-3.18.1
  export QTDIR=/usr/lib/qt-3.3
  python configure.py
  make
  make install

and configure matplotlib to use this backend::

  mkdir $HOME/.matplotlib
  cp $HOME/local/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc .matplotlib/.

and edit $HOME/.matplotlib/matplotlibrc to use the setting::

  backend      : QtAgg




