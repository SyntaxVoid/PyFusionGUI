"""Interface  for MDSplus  data  acquisition and  storage.

This  package depends  on  the MDSplus  python  package, available  from
http://www.mdsplus.org/binaries/python/

Pyfusion supports four modes for accessing MDSplus data:

 #. local
 #. thick client
 #. thin client
 #. HTTP via a H1DS MDSplus web service

The  data access  mode used  is determined  by the  mds path  and server
variables  in the  configuration file  (or supplied  to  the acquisition
class via keyword arguments)::

 [Acquisition:my_data]
 acq_class = pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition
 mydata_path = ...
 server = my.mdsdataserver.net

The  full MDSplus  node path  is  stored in  a diagnostic  configuration
section::

  [Diagnostic:my_probe]
  data_fetcher = pyfusion.acquisition.MDSPlus.fetch.MDSPlusDataFetcher
  mds_node_path = \mydata::top.probe_signal
 
Local data access
-----------------

The 'local' mode is used when a tree path definition refers to the local
file  system  rather  than  an  MDSplus  server  on  the  network.   The
:attr:`mydata_path`  entry in  the  above example  would look  something
like::

 mydata_path = /path/to/my/data


Thick client access
-------------------

The 'thick client'  mode uses an MDSplus data server  to retieve the raw
data files, but the client is responsible for evaluating expressions and
decompressing the  data. The server  tree definitions are used,  and the
server  for a  given  mds tree  is specified  by  the tree  path in  the
format::

 mydata_path = my.mdsdataserver.net::

or, if a port other than the default (8000) is used::

 mydata_path = my.mdsdataserver.net:port_number::

Thin client access
------------------

The  'thin  client' mode  maintains  a  connection  to an  MDSplus  data
server. Expressions  are evaluated and data decompressed  on the server,
requiring  greater   amounts  of  data   to  be  transferred   over  the
network. Because the thin client mode uses the tree paths defined on the
server, no path variable  is required. Instead, the :attr:`server` entry
is used::

 server = my.mdsdataserver.net

or, if a port other than the default (8000) is used::

 server = my.mdsdataserver.net:port_number


HTTP web service access
-----------------------

The  HTTP web  service  mode uses  standard  HTTP queries  via the  H1DS
RESTful API  to access the MDSplus  data. The server  is responsible for
evaluating the  data and  transmits quantisation-compressed data  to the
client over port  80.  This is especially useful if  the MDSplus data is
behind a  firewall. The  :attr:`server` attribute will  be used  for web
service access if it begins with `http://`, for example::

 server = http://h1svr.anu.edu.au/mdsplus/

The :attr:`server` attribute must be the URL component up to the MDSplus
tree    name.   In    this    example,   the    URL    for   mds    path
:attr:`\\\\h1data::top.operations.mirnov:a14_14:input_1`  and shot 58063
corresponds                                                            to
http://h1svr.anu.edu.au/mdsplus/h1data/58063/top/operations/mirnov/a14_14/input_1/


How Pyfusion chooses the access mode
------------------------------------

If an acquisition configuration  section contains a :attr:`server` entry
(which      does      not       start      with      http://),      then
:class:`~acq.MDSPlusAcquisition` will  set up a connection  to the mdsip
server when it is  instantiated. Additionally, any tree path definitions
(local and thick client) are loaded into the runtime environment at this
time. When a call to the data fetcher is made (via :meth:`getdata`), the
data  fetcher uses the  full node  path (including  tree name)  from the
configuration file.  If a matching (tree name) :attr:`_path` variable is
defined  for the  acquisition module,  then the  corresponding  local or
thick client mode will be used. If  no tree path is defined then, if the
:attr:`server` variable is defined,  pyfusion will attempt to use either
the web  services mode  (if :attr:`server` begins  with http://)  or the
thin client mode (if :attr:`server` does not begin with http://).

"""
