"""
generate flucstrucs for benchmarking
"""
import numpy as np
from pyfusion.data.tests import get_multimode_test_data, get_n_channels
from pyfusion.data.timeseries import Timebase
import cProfile, pstats, os

THIS_DIR = os.path.dirname(__file__)

stats_file = os.path.join(THIS_DIR, "stats", "fs")

n_ch = 30
n_samples = 1024*100
data = get_multimode_test_data(channels=get_n_channels(n_ch),
                               timebase = Timebase(np.arange(n_samples)*1.e-6),
                               noise = 0.1)
def do_fs():
    output_data = []
    for d in data.segment(1024):
        output_data.append(d.flucstruc())
    return output_data

cProfile.run('do_fs()', stats_file)

p = pstats.Stats(stats_file)
p.sort_stats('cum')
p.print_stats()
p.sort_stats('time')
p.print_stats()
