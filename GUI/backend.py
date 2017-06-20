# GUI/backend.py #
# John Gresl 6/19/2017 #
from collections import OrderedDict

def load_values(f):
    ## Reads a file of config values and returns a dictionary that can be used by the frontend.
    d = OrderedDict()
    with open(f) as vals:
        lines = vals.readlines()
        for line in lines:
            d[line.split(":")[0]] = line.split(":")[1].strip()
    return d

def save_values(d, f):
    ## Works inversely to load_values. Saves a dict, d, to a file, f.
    with open(f, "w") as outfile:
        out_str = "{}: {}\n"
        for key, value in d.items():
            outfile.write(out_str.format(key,value))
    return None