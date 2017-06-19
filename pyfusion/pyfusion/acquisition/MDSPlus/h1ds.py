"""Python module for the H1 data system.

This code works with python2 and python3.

It is assumed most users will be using python2, so the general
design pattern is
try:
   python2 code
except (ImportError, etc):
   python3 code

Dependencies: numpy
Optional: matplotlib (for plotting)
"""

try: # python2
    from urlparse import urlsplit, urlunsplit
except ImportError: # python3
    from urllib.parse import urlsplit, urlunsplit

try: # python2
    from urllib2 import urlopen
except ImportError: #python3
    from urllib.request import urlopen

from xml.dom import minidom
from datetime import datetime
from struct import unpack_from, calcsize

try:
    import numpy as np
except ImportError:
    print("Can't find Numpy. Please install it.")

header_dict = {'signal_min':'X-H1DS-signal-min',
               'signal_delta':'X-H1DS-signal-delta',
               'signal_units':'X-H1DS-signal-units',
               'dim_t0':'X-H1DS-dim-t0',
               'dim_delta':'X-H1DS-dim-delta',
               'dim_length':'X-H1DS-dim-length',
               'dim_units':'X-H1DS-dim-units'}
               
class H1MDSData:
    def __init__(self, shot, shottime, tree, path, data):
        self.shot = shot
        self.shottime = shottime
        self.tree = tree
        self.path = path
        self.data = data

class Signal:
    def __init__(self, signal, dim, signal_units, dim_units):
        self.signal = signal
        self.signal_units = signal_units
        self.dim = dim
        self.dim_units = dim_units

    def plot(self):
        import pylab as pl
        pl.plot(self.dim, self.signal)
        pl.xlabel(self.dim_units)
        pl.ylabel(self.signal_units)
        pl.grid(True)
        pl.show()

def add_query_to_url(url, query):
    url_parts = urlsplit(url)
    if url_parts.query == '':
        new_query = query
    else:
        new_query = '&'.join([url_parts.query, query])
    new_url_tuple = (url_parts.scheme,
                     url_parts.netloc,
                     url_parts.path,
                     new_query,
                     url_parts.fragment)

    return urlunsplit(new_url_tuple)

def signal_from_binary_url(url):
    u = urlopen(url)
    headers = u.info()
    bin_data = u.read()

    # assume little endian short (signed) integer (ref: http://docs.python.org/library/struct.html)
    # binary format details should be passed in HTTP headers....
    d = unpack_from('<%dh' %(len(bin_data)/calcsize('<h')),bin_data)

    h = {}
    try:
        for h_name, h_str in header_dict.items():
            h[h_name] = headers[h_str]
    except:
        for h_name, h_str in header_dict.items():
            h[h_name] = headers[lower(h_str)]
        
    s_arr = float(h['signal_min']) + float(h['signal_delta'])*np.array(d)
    dim_arr = float(h['dim_t0']) + float(h['dim_delta'])*np.arange(int(h['dim_length']), dtype=np.float32)
    signal = Signal(s_arr, dim_arr, h['signal_units'], h['dim_units'])
    return signal

simple_xml_value = lambda doc, tag: doc.getElementsByTagName(tag)[0].firstChild.nodeValue

def data_from_url(url):
    """Retrieve data object from H1DS URL."""
    # We use the XML view here, so make sure the URL has view=xml GET query.
    url = add_query_to_url(url, 'view=xml')

    xml_doc = minidom.parse(urlopen(url))

    shot_number   = int(simple_xml_value(xml_doc, 'shot_number'))
    shot_time_str = simple_xml_value(xml_doc, 'shot_time')
    mds_tree      = simple_xml_value(xml_doc, 'mds_tree')
    mds_path      = simple_xml_value(xml_doc, 'mds_path')

    shot_time = datetime.strptime(shot_time_str, "%d-%b-%Y %H:%M:%S.%f")
    
    data_node = xml_doc.getElementsByTagName('data')[0]
    data_node_type = data_node.getAttribute('type')

    if data_node_type == 'signal':
        data_url = data_node.firstChild.nodeValue
        data = signal_from_binary_url(data_url)
    elif data_node_type == 'scalar':
        # TODO: use proper data types
        data = float(data_node.firstChild.nodeValue)
    elif data_node_type == 'text':
        data = data_node.firstChild.nodeValue
    data_obj = H1MDSData(shot_number, shot_time, mds_tree, mds_path, data)
    return data_obj

def data_from_mds(mds_tree, mds_path, shot_number):
    query = '?shot=%(shot)d&mds-tree=%(mds_tree)s&mds-path=%(mds_path)s' %{'shot':int(shot_number),
                                                                          'mds_tree':mds_tree,
                                                                          'mds_path':mds_path}
    url = 'http://h1svr.anu.edu.au/mdsplus/request_url'+query
    xml_doc = minidom.parse(urlopen(url))
    mds_url_path = simple_xml_value(xml_doc, 'mds_url')
    mds_url = 'http://h1svr.anu.edu.au'+mds_url_path
    return data_from_url(mds_url)

#############
# Test code #
#############

def do_test():
    #test_url = 'http://h1svr/mdsplus/h1data/58623/OPERATIONS/MIRNOV/A14_14/INPUT_2/'
    #return data_from_url(test_url)
    return data_from_mds('h1data', '.operations.mirnov:a14_14:input_2', 58623)
#######################################
# Uncomment these lines to run tests: #
# From here...                        #
#######################################

if __name__ == "__main__":
    test_data = do_test()

#######################################
# To here.                            #
#######################################
