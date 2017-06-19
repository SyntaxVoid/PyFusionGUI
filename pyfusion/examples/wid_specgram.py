"""
Browse spectrograms, and has built in test function and crude window inspector.
Originally was test code to develop shot incrementing widget-like interface

David suggests Qt is better for interactive use
Advantage of this simple version is it is toolkit independent. See comments
in code.

usage:
run pyfusion/examples/wid_specgram.py shot_number=27233 diag_name='MP_SMALL'
run pyfusion/examples/wid_specgram.py NFFT=2048 shot_number=69270 diag_name='H1DTacqAxial' dev_name='H1Local' shot_list=[69270] channel_number=2
Version 6: Add flucstruc overplot and control for size of circles 
Version 5: Works nicely in ipython, two pulldowns, one for history, one
for shot selector, and selector wild card(non-blocking).  The history list only
includes shots successfully found, and either list can be navigated in order
by two down (or up) arrows, then <CR>. Now with balloon help.

Notes for pyfusion v1 version: 
   flucstruc overplot commented out
   tricky options for shot selector list not implemented - just shot_list
   show signals button not yet implemented
Needs process_cmd_line_args.py, in python path (e.g. ~/python dir)
"""

""" Comments on code development:
Notes:
Need to include channel selector - hardwired to mirnov 8 at the moment.  
Initially will use radio button channel selector - one at a time.
Ultimate plan is to include an SQL query box to narrow range of shots
Should really try to modularise this
Display updates are 90% right now - still one step behind in adjusting FFT params

Initial addition of Tix interface, creates a shot "box" with pulldowns
Tix can be tested using test_Tix_widgets.py (boyd)
Only successful with tkagg - otherwise blocks or needs root.mainloop()
With gtkagg, sort of works, but doesn't update shot, and blocks
"""

from matplotlib.widgets import RadioButtons, Button
import pylab as pl
from numpy import sin, pi, ones, hanning, hamming, bartlett, kaiser, arange, blackman, cos, sqrt, log10, fft

import pyfusion
try:
    pyfusion.VERBOSE
except:
    pyfusion.VERBOSE=int(pyfusion.config.get(
        'global', 'verbose',vars={'verbose': '1'}))

# local definitions for a few windows. mlab.windows are defined
# differently, and the returned value is multiplied by the input data
# already.  There are only two of the mlab.window_hanning and
# mlab.window_none.  However, to be useful to David's function, they
# need to be exported I think.

def local_none(vec):
    return(ones(len(vec)))

def local_hanning(vec):
    return(hanning(len(vec)))

def local_hamming(vec):
    return(hamming(len(vec)))

def local_blackman(vec):
    return(blackman(len(vec)))

def local_bartlett(vec):
    return(bartlett(len(vec)))

# not sure about this, but it is pretty right.
def local_kaiser3(vec):
    return(kaiser(len(vec),3*pi))


def local_wider(vec):
    """ Flat top in middle, cos at edges - meant to be narrower in f
    but not as good in the wings
    """
    N=len(vec)
    k=arange(N)
    w = sqrt(sqrt(1 - cos(2*pi*k/(N-1))))
#    w = (1 - 1.93*cos(2*pi*k/(N-1)) + 1.29*cos(4*pi*k/(N-1)) 
#         -0.388*cos(6*pi*k/(N-1)) +0.032*cos(8*pi*k/(N-1)))
    return(w)

def local_flat_top_freq(vec):
    N=len(vec)
    k=arange(N)
    w = (1 - 1.93*cos(2*pi*k/(N-1)) + 1.29*cos(4*pi*k/(N-1)) 
         -0.388*cos(6*pi*k/(N-1)) +0.032*cos(8*pi*k/(N-1)))
    return(w)

global shot_number, shot_list, channel_number, chan_name, marker_size, wild_card, diag_name

def get_local_shot_numbers(partial_name):
    """ This used to be in utils.  For now replace by shot_list variable
    Probably good to keep for locals, and systems that have an easily accessible
    shot list.  But how to deal with a list of 100,000 shots?
    """
    global shot_list
    return(shot_list)

# defaults
wild_card = ''

dev_name= 'LHD' # 'H1Local'

shot_number=None
channel_number=None

diag_name=""
shot_list=[]
cmap=None
#xextent=None  # was here, really belongs in data.spectrogram
NFFT=512
Fsamp=2
Fcentre=0
marker_size=0
detrend=pl.detrend_none
_window = local_wider # none causes math errors in specgram sometimes 
foverlap=0.75   # 0 is the cheapest, but 3/4 looks better
_type='F'
fmod=0
# t_max=0.08

execfile('process_cmd_line_args.py')

device = pyfusion.getDevice(dev_name)

chan_name=''

if dev_name=='TestDevice':
    chan_name='testch1'
    shot_number=1000
elif (dev_name=='H1') or(dev_name=='H1Local'):
    chan_name='mirnov_1_8'
    shot_number=58123
elif dev_name=='HeliotronJ': 
    chan_name='MP1'
    shot_number=33911
elif dev_name=='LHD': 
    shot_list = [27233,36150,90091]
    diag_name='MP_SMALL'
    chan_name='HMP01'
    shot_number=90091
elif dev_name=='JT60U': 
    chan_name='PMP01'
    shot_number=46066
elif dev_name=='TJII': 
    chan_name='mirnov_5p_105'
    shot_number=18991
else:  # need something so process_cmd can work
    chan_name=''
    shot_number=0
if wild_card == '': wild_card = chan_name+'*'    
if pyfusion.VERBOSE>2: 
    print("Using device '%s', chan_name '%s', shot_number %d" %    
          (dev_name, chan_name, shot_number))

if channel_number==None: channel_number=0

# tweak above parameters according to command line args
execfile('process_cmd_line_args.py')

# arrays for test signal
tm=arange(0,0.02,1e-6)
y=sin((2e5 + 5e3*sin(fmod*2*pi*tm))*2*pi*tm)

def call_spec():
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod, chan_name, diag_name
    print len(y), NFFT,foverlap, _type, fmod
    ax = pl.subplot(111)
    z=_window(y)
    if _type=='F': 
        shot=callback.get_shot()
        print("shot=%d") % shot
        data = device.acq.getdata(shot, diag_name)    
        if chan_name=='':
            try:
                ch=data.channels
                print("Choosing from", [chn.name for chn in ch])
                name=ch[channel_number].name
            except:
                print "Failed to open channel database - try mirnov_1_8"
                name='mirnov_1_8'
                name='mirnov_linear_2'
        else:        
            name=chan_name

#        data = pyfusion.load_channel(shot,name)
#        data = pyfusion.acq.getdata(shot_number, diag_name)    
        if data==None: return(False)
        
        if _window==local_none: windowfn=pl.window_none
#        else: windowfn=pl.window_hanning
        elif _window==local_hanning: windowfn=pl.window_hanning
        else: windowfn=_window(arange(NFFT))
        clim=(-60,20)   # eventually make this adjustable
# colorbar commented out because it keeps adding itself
        data.plot_spectrogram(NFFT=NFFT, windowfn=windowfn, noverlap=foverlap*NFFT, 
                              channel_number=channel_number)
#                         colorbar=True, clim=clim)
#        colorbar() # used to come up on a separate page, fixed, but a little clunky - leave for now

        return(True)
    elif _type == 'T':
# some matplotlib versions don't know about Fc
        pl.specgram(z*y, NFFT=NFFT, Fs=Fsamp, detrend=detrend,
#                 window = _window
                 noverlap=foverlap*NFFT, cmap=cmap)
    elif _type == 'L':
        pl.plot(20*log10(abs(fft.fft(y*z))))
    elif _type == 'W':
        pl.plot(z)
    elif _type =='C':
        pl.plot(hold=0)
    else: raise ' unknown plot type "' + _type +'"'
#    pl.show()

# ------  END of call_spec

oldinter = pl.isinteractive
pl.ioff()

ax = pl.subplot(111)
pl.subplots_adjust(left=0.25)
pl.subplots_adjust(right=0.95)  # see also the colorbar params in core.py
#call_spec()

#Buttons Start Here

bxl=0.02
bw=0.12  # width (for most)
axcolor = 'lightgoldenrodyellow'

#define the box where the buttons live
rax = pl.axes([bxl, 0.87, bxl+bw, 0.11], axisbg=axcolor)
radio = RadioButtons(rax, ('no marker',  '40', '80', '120'),active=0)
def msfunc(label):
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod, marker_size
    msdict = {'no marker':0, '40':40, '80':80, '120':120}
    marker_size = msdict[label]
    print("marker_size", marker_size)
    callback.redraw()   # really should add markers here! (this is a call without getting new data)

radio.on_clicked(msfunc)


rax = pl.axes([bxl, 0.68, bxl+bw, 0.18], axisbg=axcolor)
radio = RadioButtons(rax, ('win 128', '256', '512', '1024','2048','4096'),active=2)
def hzfunc(label):
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod
    hzdict = {'win 128':128, '256':256, '512':512, '1024':1024,
              '2048':2048, '4096':4096}
    NFFT = hzdict[label]
    call_spec()

radio.on_clicked(hzfunc)


rax = pl.axes([bxl, 0.48, bxl+bw, 0.19], axisbg=axcolor)
radio = RadioButtons(rax, ('overlap 0', '1/4', '1/2', '3/4','7/8','15/16'),active=3)

def ovlfunc(label):
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod
    ovldict = {'overlap 0':0, '1/4':0.25, '1/2':0.5, '3/4':0.75, '7/8':0.875,
               '15/16':0.9375}
    foverlap = ovldict[label]
    call_spec()

radio.on_clicked(ovlfunc)

rax = pl.axes([bxl, 0.23, bxl+bw, 0.24], axisbg=axcolor)
radio = RadioButtons(rax, ('no window',  'Wider', 'Bartlett','Hamming', 'Hanning',
                           'Blackman', 'Kaiser3','Flat-top-F'), active=1)
def winfunc(label):
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod
    windict = {'no window':local_none, 'Hanning':local_hanning, 
               'Wider': local_wider,
               'Hamming':local_hamming, 'Blackman':local_blackman, 
               'Bartlett':local_bartlett, 'Kaiser3':local_kaiser3,
               'Flat-top-F':local_flat_top_freq}
    _window = windict[label]
    call_spec()

radio.on_clicked(winfunc)

rax = pl.axes([bxl, 0.08, bxl+bw, 0.14], axisbg=axcolor)
radio = RadioButtons(rax, ('f-t plot', 'test data', 'log-spect', 'window', 'clear'))
def typfunc(label):
    global y,NFFT,Fsamp,Fcentre,foverlap,detrend,_window, _type, fmod
    typdict = {'f-t plot':'F', 'test data':'T', 'log-spect':'L', 'window':'W', 'clear':'C'}
    _type = typdict[label]
    call_spec()

radio.on_clicked(typfunc)
##############################################################
# This line is where I joined the radio button code to the shot number code
# Would be nice to pull this apart into two modules and a short script.
###############################################################
#
#ax = subplot(111)
#
#subplots_adjust(left=0.3)

x0=0
y0=0.02

try:
    import Tix
    HaveTix=True
except:
    print("Tix module not available: shot button inactive")
    HaveTix=False


# before putting these in a module, check that exec works on module vars
def make_inherited_var(name):
    exec('val='+name)
    return("%s=%s" % (name, val))

def inherited_vars(vars=None, extras=None):
    """ Return a list of var=value suitable forprocess_cmd_line_args.py
    vars defaults to a safe, comprehensive set from pyfusion.settings
    extras as those particular to some routines.
    """
    if vars == None: vars=['pyfusion.settings.SHOT_T_MIN',
                           'pyfusion.settings.SHOT_T_MAX']
    lst = []
    if extras != None:
        for var in extras: 
            lst += [var]
    for var in vars:
        lst += [make_inherited_var(var)]
    return(lst)    

class IntegerCtl:
    """ provides an environment for button on_clicked functions to
    share variables with each other and plotting routines, rather than
    trying to access everythin back through the events passed.
    """
# these maybe should be in an init, but this is OK python code.
    global shot_number
    shot=shot_number

    def set_shot(s):
        shots

    def get_shot(self):
        return(self.shot)
    
# probably need to redraw the whole graph
    def redraw(self):
        global hist_box, HaveTix, marker_size
        bshot.label.set_text(str(self.shot))
        status=call_spec()
        if HaveTix:  # update shot field in either case, only update history if good
            # this updates hist_box if the shot was changed by the other (matplotlib) widgets
            hist_box.set_silent(str(self.shot))
            if status==True:
                hist_box.add_history(str(self.shot))

        print("marker_size", marker_size)
#        if marker_size>0: plot_flucstrucs_for_shot(self.shot, size_factor=marker_size, savefile='')
#        pl.draw()  # what does this do?
        return(status) # False if no data

    def frew(self, event):
        self.shot -= 10
        self.redraw()

    def rew(self, event):
        self.shot -= 1
        self.redraw()

    def fwd(self, event):
        self.shot += 1
        self.redraw()

    def ffwd(self, event):
        self.shot += 10
        self.redraw()

# extra fast fwd is 100+
    def Xffwd(self, event):
        self.shot += 100
        self.redraw()

    def Xfrew(self, event):
        self.shot -= 100
        self.redraw()

    def wid_specgram(self, event):
        import os 
        args = ['python',  'examples/Boyds/wid_specgram.py']
        args += inherited_vars()
# need to pass a string array otherwise treated as array of chars!
        args += [str('shot_number=%d' % (self.shot))]
        if pyfusion.VERBOSE>5: 
            print("args to spawn", args)
            print("")  # so we can see the output
        os.spawnvp(os.P_NOWAIT,'python', args)
        self.redraw()

    def wid_showsigs(self, event):
        import os 
        args = ['python',  'examples/Boyds/wid_showsigs.py']
        args += inherited_vars()
# need to pass a string array otherwise treated as array of chars!
        args += [str('shot_number=%d' % (self.shot))]
        if pyfusion.VERBOSE>5: 
            print("args to spawn", args)
            print("")  # so we can see the output

        os.spawnvp(os.P_NOWAIT,'python', args)
        self.redraw()

#-------- End of class IntegerCtl:

if HaveTix:
## This is intialization code, even though indented (it is conditional)
#    from pyfusion.utils import get_local_shot_numbers
#    from pyfusion.datamining.clustering.plots import plot_flucstrucs_for_shot
    print('Special import for HaveTix')
    global shot_string, select_string, do_shot, hist_box, wild_box, select_list, select_box, wild_string

# this has to be before Tix.StringVar() is called in ubuntu 10.04
    root=Tix.Tk(className='ShotSelect')
    select_list=[]
    wild_string= Tix.StringVar()
    shot_string= Tix.StringVar()
    select_string= Tix.StringVar()
    def do_shot(sstr=None):
        print('sstr=', sstr)
        if sstr == '': 
            print('No shot defined - is this a blocking problem? - default to 1')
            sstr='1'
        callback.shot=int(sstr)
        if callback.redraw():
            print('success')
        else: print('error')
            
    def update_select(partial_name=None):
        global select_box, select_list, wild_box
        print("update select, but doesn't work?")
        # put the name in the wildcard entry area in case we are called from other than the command
        wild_box.set_silent(partial_name)

        if partial_name.find('(') >= 0:  # a function call
            print('executing' + partial_name)
               # does this exec in the local context?
            exec('select_list='+partial_name)

        elif partial_name.find('/') >= 0:  # a file list
               # run the file through an awk filter to get the first "word"
               # on each line, which should be a shot number
            pass

        elif partial_name.find('*') >= 0:   
            # really should use a regexp, but then have to get the number part
            select_list=get_local_shot_numbers(
                partial_name=string.strip(partial_name,'*')) # get a new list
         
        if len(select_list)==0: select_list= ['none?']  # put 'none there if it failed
        else: 
            for s in select_list: select_box.insert(Tix.END, s) # put in widget list if OK
        select_box.set_silent(select_list[0])        # and put the top entry in the window

    def clear_select():  # doesn't work!
        global select_list
        select_list=['none']
        update_select('')
        print ('clear', len(select_list))

    def ShotWid():
        """ this simple widget accepts a shot and sets the current one
        It is a function in the IntegerCtl class, so it communicates with
        its vars easily and calls do_shot to update the shot.  THe
        shot pulldown stops working in python (ordinary) after 1
        pulldown?

        """
        global hist_box, select_box, wild_box
#        root=Tix.Tk(className='ShotSelect')  # was here but needs to
#        be in effect before Tix.StringVar() is called
        top = Tix.Frame(root, bd=1, relief=Tix.RAISED)
        hist_box=Tix.ComboBox(top, label="Shot", editable=True, history=True,
                             variable=shot_string, command=do_shot,
                             options='entry.width 8 listbox.height 10 ')
        hist_box.pack(side=Tix.TOP, anchor=Tix.W)
        hist_box.set_silent('33373')
        hist_balloon=Tix.Balloon(top)
        hist_balloon.bind_widget(hist_box, balloonmsg='Choose or enter shot number, valid ones are saved here')

        wild_box=Tix.ComboBox(top, label="Filter", editable=1, history=1,
                             variable=wild_string, command=update_select,
                             options='entry.width 20 listbox.height 5 ')   # allow room for expressions
        wild_box.pack(side=Tix.TOP, anchor=Tix.W)
        wild_balloon=Tix.Balloon(top)
        wild_balloon.bind_widget(wild_box, 
                                 balloonmsg='Choose or enter new filter in one of three forms,' + 
                                 'a Python expression (must have () or []), '+ 
                                 'a directory specification including a * or ' +
                                 'the name of a file containing lines beginning with a shot number. '
                                 'Results can be chosen using "Filtered Shots"')

        select_box=Tix.ComboBox(top, label="Filtered Shots", history=False,
                             variable=select_string, command=do_shot,
                             options='entry.width 8 listbox.height 40 ')
        btn = Tix.Button(select_box, text='Clear',command=clear_select)
        btn.pack(anchor=Tix.CENTER)
        select_box.pack(side=Tix.TOP, anchor=Tix.W)
        select_balloon=Tix.Balloon(top)
        select_balloon.bind_widget(select_box, balloonmsg='pull down to find a shot selected by "Filter""')
        #wild_box.set_silent('MP1')  # not silent - want it all to happen, but setvar doesn't work

        update_select(partial_name=wild_card)

        top.pack(side=Tix.TOP, fill=Tix.BOTH, expand=1)
# no need in pylab provided tkagg is used     root.mainloop()
# in fact, may conflict and block - hard to sort out what blocks and when, why


callback = IntegerCtl()
but_h = 0.045

global button_layout_cursor

def mybut(text, dummy, xl, yb, xw=0, yh=0, axisbg=None, color=0.85, fun=None, bspace=0.005):
    """ create axes and populate button with text, automatically adjusting
    xw if not given.  Has a side effect on xl. (button_layout_cursor)
    dummy is for if and when I can place these on an obect rather than using pylab
    """
    if axisbg==None: axisbg='lightgoldenrodyellow'

    global button_layout_cursor
    if xw==0: xw=0.015*(len(text)+1)
    if yh==0: yh=0.05
##    thisax=fig.add_axes([xl, yb, xw, yh], axisbg=axisbg) fundamentally wrong
    thisax=pl.axes([xl, yb, xw, yh], axisbg=axisbg)
    thisbut=Button(thisax, text)
    thisbut.on_clicked(fun)
    button_layout_cursor += xw+bspace
    return(thisbut)

button_layout_cursor=0.01

fig=0

spectest=mybut('specgram', fig, button_layout_cursor, y0, 0, but_h, fun=callback.wid_specgram, axisbg='yellow')
sigstest=mybut('showsigs', fig, button_layout_cursor, y0, 0, but_h, fun=callback.wid_showsigs)
bXfrew=mybut('<<<', fig, button_layout_cursor, y0, 0, but_h, fun=callback.Xfrew)
bfrew=mybut('<<', fig, button_layout_cursor, y0, 0, but_h, fun=callback.frew)
brew=mybut('<', fig, button_layout_cursor, y0, 0, but_h, fun=callback.rew)
bshot=mybut('12345', fig, button_layout_cursor, y0, 0, but_h, fun=callback.shot)
bfwd=mybut('>', fig, button_layout_cursor, y0, 0, but_h, fun=callback.fwd)
bffwd=mybut('>>', fig, button_layout_cursor, y0, 0, but_h, fun=callback.ffwd)
bXffwd=mybut('>>>', fig, button_layout_cursor, y0, 0, but_h, fun=callback.Xffwd)

if oldinter: pl.ion()

# this is sort of initialisation code, but needed to be in a function
if HaveTix:  ShotWid()

callback.redraw()

pl.show()

