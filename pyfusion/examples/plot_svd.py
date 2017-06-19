""" plot the svd of a diag, either one with arbitrary time bounds
or the sequence of svds of numpts starting at start_time (sec)
keys typed at the terminal allow stepping backward and forward.

Typical usage : run  dm/plot_svd.py start_time=0.01 "normalise='v'" use_getch=0
"""

def order_fs(fs_set, by='p'):
    """ Dave's code returns an unordered set - need to order by singular value (desc)
    """
    fsarr_unsort=[]
    for fs in fs_set: fsarr_unsort.append(fs)
    if by == 'p': revord = argsort([fs.p for fs in fsarr_unsort ])
    else: raise ValueError, str(" sort order %s not supported " % by)
    fs_arr = []
    for ind in revord: fs_arr.append(fsarr_unsort[ind])
    fs_arr.reverse() # in place!
    return(fs_arr)


import subprocess, sys, warnings
from numpy import sqrt, argsort, average, mean, pi
import pyfusion as pf
import pyfusion.utils
import pylab as pl
import numpy as np


try:
    import getch
    use_getch = True
except:
    use_getch = False
print(" getch is %savailable" % (['not ', ''][use_getch]))

dev_name='H1Local'   # 'LHD'
#dev_name='LHD'
if dev_name == 'LHD': 
    diag_name= 'MP'
    shot_number = 27233
    #shot_range = range(90090, 90110)
elif dev_name == 'H1Local': 
    diag_name = "H1DTacqAxial"
    shot_number = 69270

hold=0
exception=Exception
time_range = None
channel_number=0
start_time = None
numpts = 512
normalise='0'
separate=1
verbose=0
max_fs = 2

device = pf.getDevice(dev_name)

try:
    old_shot
except:
    old_shot=0

#execfile('process_cmd_line_args.py')
exec(pf.utils.process_cmd_line_args())

print(" %s using getch" % (['not', 'yes, '][use_getch]))
if use_getch: print('plots most likely will be suppressed - sad!')

if old_shot>0: # we can expect the variables to be still around, run with -i
    if (old_diag != diag_name) or (old_shot != shot_number): old_shot=0

if old_shot == 0: 
    d = device.acq.getdata(shot_number, diag_name) # ~ 50MB for 6ch 1MS. (27233MP)
    old_shot = shot_number
    old_diag = diag_name

if time_range != None:
    d.reduce_time(time_range)

if start_time == None:
    sv = d.svd()
    sv.svdplot(hold=hold)

else:
    # first copy the segments into a list, so they can be addressed
    # this doesn't seem to take up much extra memory.
    segs=[]
    for seg in d.segment(numpts):
        segs.append(seg)
    starts = [seg.timebase[0] for seg in segs]
    ord_segs=[]
#    for ii in argsort(starts):
    i=0
    argsrt = argsort(starts)
    while i < len(starts):
        ii = argsrt[i]
        seg=segs[ii]
        if seg.timebase[0] > start_time: 
#            print("normalise = %s" % normalise)
            if (normalise != 0) and (normalise != '0'): 
                seg.subtract_mean().normalise(normalise,separate).svd().svdplot()
            else: 
                seg.subtract_mean().svd().svdplot()
            try:
                if seg.scales != None:
                    fig=pl.gcf()
                    oldtop=fig.subplotpars.top
                    fig.subplots_adjust(top=0.78)
                    ax=pl.subplot(8,2,-2)
                    xticks = range(len(seg.scales))
                    pl.bar(xticks, seg.scales, align='center')
                    ax.set_xticks(xticks)
                    # still confused - sometimes the channels are the names bdb
                    try:
                        seg.channels[0].name
                        names = [sgch.name for sgch in seg.channels]
                    except:
                        names = seg.channels

                    short_names,p,s = pf.data.plots.split_names(names)
                    short_names[0]=seg.channels[0].name  # first one in full
                    ax.set_xticklabels(short_names)
                    ax.set_yticks(ax.get_ylim())
# restoring it cancels the visible effect - if we restore, it should be on exit
#                    fig.subplots_adjust(top=oldtop)
            except None:        
                pass
            pl.suptitle("Shot %s, t_mid=%.5g, norm=%s, sep=%d" % 
                        (shot_number, average(seg.timebase),
                         normalise, separate))

            fs_set=seg.flucstruc(method=0, separate=separate)
            fs_arr = order_fs(fs_set)
            for fs in fs_arr[0:min([len(fs_arr)-1,max_fs])]:
                RMS_scale=sqrt(mean(seg.scales**2))
                print("amp=%.3g:" % (sqrt(fs.p)*RMS_scale)),

                print("f=%.3gkHz, t=%.3g, p=%.2f, a12=%.2f, E=%.2g, adjE=%.2g, %s" %
                      (fs.freq/1000, fs.t0, fs.p, fs.a12, fs.E,
                       fs.p*fs.E*RMS_scale**2, fs.svs()))
# first fs?        
            fs=fs_arr[0]    
            ax=pl.subplot(8,2,-3)
            fs.fsplot_phase()    
            pl.ylim([-np.pi,np.pi])
            if use_getch: 
                pl.show()
                k=getch.getch()
            else: k=raw_input('enter one of "npqegsS" (return->next time segment)')
            if k=='': k='n'
            if k in 'bBpP': i-=1
            # Note - if normalise or separate is toggled, it doesn't
            #    affect segs already done.
            elif k in 'tT': 
                if (normalise==0) or (normalise=='0'): normalise ='rms'
            elif k in 'qQeE':i=999999
            elif k in 'gG': use_getch=not(use_getch)
            elif k in 'S': separate=not(separate)
            elif k in 's': # plot the signals in a new frame
                pl.figure()
                segs[ii].plot_signals()
                pl.figure(1)
            else:  i+=1
            if verbose: print i,ii
        else: i+=1

        
