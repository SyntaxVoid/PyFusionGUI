# Utilities/jtools.py #
# John Gresl 6/19/2017 #
import re
import numpy as np
from pyfusion import DEFAULT_CONFIG_FILE

def valid_int_from_str(s):
    # Returns true if s can be converted to an integer.
    try:
        int(s)
        return 1
    except:
        return 0


def valid_float_from_str(s):
    try:
        float(s)
        return 1
    except:
        return 0


def complex_mag(z):
    # Returns the magnitude sqrt(Re(z)^2 + Im(z)^2) of a numpy.complex number, z
    return np.sqrt(z.real**2 + z.imag**2)

def complex_mag_list(zz):
    # Takes in an array of complex numbers and returns an array of their magnitudes
    return [complex_mag(z) for z in zz]


def type_verify(var, typ):
    return type(var) == typ


def scan_config(f):
    # Scans a config file, f, and returns all diagnostics in a list.
    diagnostics = []
    with open(f) as conf:
        conflines = conf.readlines()
        for line in conflines:
            if line.startswith("[Diagnostic:"):
                temp = list(line[12:])
                temp.remove("]")
                diagnostics.append("".join(temp).strip())
    return diagnostics


def valid_probe_array(s, f = DEFAULT_CONFIG_FILE):
    # Checks the string s to ensure it is a valid probe array pointname from the pyfusion config file, f.
    return s in scan_config(f)


def return_methods():
    return ["k_means", "EM_VMM", "EM_GMM"]


def valid_method(s):
    # Returns true if s is a valid method. Methods are hard coded below... Fix this?
    return s in return_methods()


def valid_window(s):
    # Returns true if s is a valid time window in the format "t0-t1" where t1>t0
    m = re.compile("^\d+ *- *\d+$")  # Checks format
    if m.match(s) is not None:
        if int(s.split("-")[0]) > int(s.split("-")[1]):  # Checks t1>t0
            return 0
        return 1
    return 0


def time_window_parser(s):
    if valid_window(s):
        return [int(s.split("-")[0].strip()), int(s.split("-")[1].strip())]
    return None

def valid_num_or_range(s):
    # Returns true if s is either a number or two numbers separated by a "-"
    s = s.strip()
    m1 = re.compile("^\d*$")
    m2 = re.compile("^\d+ *- *\d+$")
    if m1.match(s) is not None:
        return 1
    if m2.match(s) is not None:
        if int(s.split("-")[0]) >= int(s.split("-")[1]):
            print("Wrong Format: ({})".format(s))
            return 0
        return 1
    print("No Match: ({})".format(s))
    return 0

def valid_shot_str(s):
    segments = s.split(",")
    for segment in segments:
        temp = segment.strip()
        m1 = re.compile("^\d+$")
        m2 = re.compile("^\d+ *- *\d+$")
        if m1.match(temp): pass
        elif m2.match(temp):
            if int(temp.split("-")[0]) >= int(temp.split("-")[1]): return 0
        else: return 0
    return 1


def shot_str_parser(s):
    # Parses a string and converts it into a list based on rules.
    # Example: An input of "1,6-9,12" will return [1,6,7,8,9,12]
    out = []
    segments = s.split(",")
    for segment in segments:
        # Verify each segment is either a single value or a range with one "-"
        temp = segment.strip()
        if valid_num_or_range(temp):
            m1 = re.compile("^\d+$")
            m2 = re.compile("^\d+ *- *\d+$")
            if m1.match(temp):
                out.append(int(temp))
            elif m2.match(temp):
                out += range(int(temp.split("-")[0]), int(temp.split("-")[1])+1)
            else:
                print("???????")
        else:
            raise ValueError("Invalid string.")
    return out


def squareish_grid(n, swapxy=False):
    # Returns the grid closest to a square that will fit n elements
    rows = 1
    cols = 1
    while rows*cols < n:
        if rows == cols:
            cols += 1
        else:
            rows += 1
    return (rows, cols) if not swapxy else (cols, rows)


def find_closest(arr, x):
    m = abs(arr[0]-x)
    x0 = arr[0]
    n0 = 0
    for n, item in enumerate(arr):
        if abs(item-x) < m:
            m = abs(item-x)
            x0 = item
            n0 = n
    return n0, x0
