""" simple example to plot a spectrogram, uses command line arguments

    run pyfusion/examples/plot_specgram shot_number=69270

    See process_cmd_line_args.py
"""

import pyfusion as pf
import pylab as pl

dev_name='H1Local'   # 'LHD'
device = pf.getDevice(dev_name)

#shot_number = 27233
shot_number = 69270

diag_name= 'MP'
diag_name = "H1DTacqAxial"
time_range = None
channel_number=0
hold=0

# ideally should be a direct call, passing the local dictionary
import pyfusion.utils
exec(pf.utils.process_cmd_line_args())
#execfile('process_cmd_line_args.py')

d = device.acq.getdata(shot_number, diag_name)
if time_range != None:
    d.reduce_time(time_range)

d.subtract_mean().plot_spectrogram(channel_number=channel_number, hold=hold)
