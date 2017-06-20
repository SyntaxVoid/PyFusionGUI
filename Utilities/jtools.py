# Utilities/jtools.py #
# John Gresl 6/19/2017 #
import re
import numpy as np



def complex_mag(z):
    # Returns the magnitude sqrt(Re(z)^2 + Im(z)^2) of a numpy.complex number, z
    return np.sqrt(z.real**2 + z.imag**2)

def complex_mag_list(zz):
    # Takes in an array of complex numbers and returns an array of their magnitudes
    return [complex_mag(z) for z in zz]


def type_verify(var, typ):
    return type(var) == typ


def valid_num_or_range(s):
    # Returns true if s is either a number or two numbers separated by a "-"
    m1 = re.compile("^\d*$")
    m2 = re.compile("^\d*-\d*$")
    if m1.match(s) is not None:
        return 1
    if m2.match(s) is not None:
        if int(s.split("-")[0]) >= int(s.split("-")[1]):
            return 0
        return 1
    return 0


def shot_str_parser(s):
    # Parses a string and converts it into a list based on rules.
    # Example: An input of "1,6-9,12" will return [1,6,7,8,9,12]
    if not type_verify(s, str):
        raise TypeError("s must be type str. got type {}".format(type(s)))
    out = []
    segments = s.split(",")
    for segment in segments:
        # Verify each segment is either a single value or a range with one "-"
        if valid_num_or_range(segment):
            m1 = re.compile("^\d+$")
            m2 = re.compile("^\d+-\d+$")
            if m1.match(segment):
                out.append(int(segment))
            elif m2.match(segment):
                out += range(int(segment.split("-")[0]), int(segment.split("-")[1])+1)
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
