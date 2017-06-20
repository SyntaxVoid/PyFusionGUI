### Utilities/jtools.py ###
### John Gresl 6/19/2017 ###
import re

def type_verify(var,typ):
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
    out = [ ]
    segments = s.split(",")
    for segment in segments:
        # Verify each segment is either a single value or a range with one "-"
        if valid_num_or_range(segment):
            m1 = re.compile("^\d+$")
            m2 = re.compile("^\d+-\d+$")
            if m1.match(segment):
                out.append(int(segment))
            elif m2.match(segment):
                out += range(int(segment.split("-")[0]),int(segment.split("-")[1])+1)
        else:
            raise ValueError("Invalid string.")
    return out
