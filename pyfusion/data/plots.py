"""
Note, plots (this file) doesn't have unittests
"""
from matplotlib.widgets import CheckButtons
import pylab as pl

import numpy as np

from pyfusion.data.utils import peak_freq, split_names
#can't from pyfusion.config import get
import pyfusion

plot_reg = {}

# the registration function is similar but separate for plots and filters
def register(*class_names):
    def reg_item(plot_method):
        for cl_name in class_names:
            if not plot_reg.has_key(cl_name):
                plot_reg[cl_name] = [plot_method]
            else:
                plot_reg[cl_name].append(plot_method)
        return plot_method
    return reg_item

@register("TimeseriesData")
def plot_signals(input_data, filename=None,downsamplefactor=1,n_columns=1):
    import pylab as pl
    n_rows = input_data.signal.n_channels()
    n_rows = int(n_rows/n_columns)
    print str(n_rows) + ' ' + str(n_columns)
    for row in range(n_rows):
        for col in range(n_columns):
            print (row)*n_columns+col+1
            pl.subplot(n_rows, n_columns, row*n_columns+col+1)
            if downsamplefactor==1:
                pl.plot(input_data.timebase, input_data.signal.get_channel(row*n_columns+col))
                pl.axis([-0.01,0.1,-5, 5])
            else:
                plotdata=input_data.signal.get_channel(row*n_columns+col)
                timedata=input_data.timebase
                pl.plot(timedata[0:len(timedata):downsamplefactor], plotdata[0:len(timedata):downsamplefactor])
                pl.axis([-0.01,0.1,-5,5])
    if filename != None:
        pl.savefig(filename)
    else:
        pl.show()

@register("TimeseriesData")
def plot_spectrogram(input_data, windowfn=None, channel_number=0, filename=None, coloraxis=None, noverlap=0,NFFT=None, **kwargs):
    import pylab as pl
    
    if windowfn == None: windowfn=pl.window_hanning

    # look in the config file section Plots for NFFT = 1234
    # Dave - how about a method to allow this in one line
    # e.g. pyfusion.config.numgetdef('Plots','NFFT', 2048)
    # usage:  
    # if (NFFT==None): NFFT = pyfusion.config.numgetdef('Plots','NFFT', 2048)
    # 
    # also nice to have pyfusion.config.re-read()
    if NFFT == None:
        try:
            NFFT=(eval(pyfusion.config.get('Plots','NFFT')))
        except:
            NFFT = 2048

    pl.specgram(input_data.signal.get_channel(channel_number), NFFT=NFFT, noverlap=noverlap, Fs=input_data.timebase.sample_freq, window=windowfn, **kwargs)
    #accept multi or single channel data (I think?)
        
    if coloraxis != None: pl.clim(coloraxis)
    else:
        try:
            pl.clim(eval(pyfusion.config.get('Plots','coloraxis')))
        except:
            pass

    # look in the config file section Plots for a string like FT_Axis = [0,0.08,0,500]
    # don't quote
    try:
        pl.axis(eval(pyfusion.config.get('Plots','FT_Axis')))
    except:
        pass

    try:
        pl.title("%d, %s"%(input_data.meta['shot'], input_data.channels[channel_number].name))
    except:
        pl.title("%d, %s"%(input_data.meta['shot'], input_data.channels.name))
        
    if filename != None:
        pl.savefig(filename)
    else:
        pl.show()

#def plot_multichannel_coord(input_data, coord=None, savefig=None):
#    pass


def join_ends(inarray,add_2pi = False,add_360deg=False,add_lenarray=False,add_one=False):
    """used in old code, needs clean up...."""
    output = np.resize(inarray,(len(inarray)+1,))
    if add_2pi:
        output[-1] = output[-1]+2*pi
    elif add_360deg:
        output[-1] = output[-1]+360.0
    elif add_lenarray:
        output[-1] = output[-1]+len(inarray)+1
    elif add_one:
        output[-1] = output[-2]+1
    return output

def posNegFill(x,y1,y2):
    diff = y2 - y1
    pos = []
    neg = []
    xx1 = [x[0]]
    xx2 = [x[0]]
    yy1 = [y1[0]]
    yy2 = [y2[0]]
    oldSign = (diff[0] < 0 )
    npts = x.shape[0]
    for i in range(1,npts):
            newSign = (diff[i] < 0)
            if newSign != oldSign:
                    xz,yz = findZero(i,x,y1,y2)
                    xx1.append(xz)
                    yy1.append(yz)
                    xx2.reverse()
                    xx1.extend(xx2)
                    yy2.reverse()
                    yy1.extend(yy2)
                    if oldSign:
                            neg.append( (xx1,yy1) )
                    else:
                            pos.append( (xx1,yy1) )
                    xx1 = [xz,x[i]]
                    xx2 = [xz,x[i]]
                    yy1 = [yz,y1[i]]
                    yy2 = [yz,y2[i]]
                    oldSign = newSign
            else:
                    xx1.append( x[i])
                    xx2.append( x[i])
                    yy1.append(y1[i])
                    yy2.append(y2[i])
                    if i == npts-1:
                            xx2.reverse()
                            xx1.extend(xx2)
                            yy2.reverse()
                            yy1.extend(yy2)
                            if oldSign :
                                    neg.append( (xx1,yy1) )
                            else:
                                    pos.append( (xx1,yy1) )
    return pos,neg

class Energy:
    def __init__(self,energy_list,initial_list):
            self.value = 0.0
            self.energy_list = energy_list
            for i in range(len(initial_list)):
                    if initial_list[i]:
                            self.value += self.energy_list[i]
    def add(self,elmt):
            self.value += self.energy_list[elmt]
    def sub(self,elmt):
            self.value -= self.energy_list[elmt]

def findZero(i,x,y1,y2):
    im1 = i-1
    #print y1[i],y1[im1],x[i],x[im1],(x[i]-x[im1])
    m1 = (y1[i] - y1[im1])/(x[i] - x[im1])
    m2 = (y2[i] - y2[im1])/(x[i] - x[im1])
    b1 = y1[im1] - m1*x[im1]
    b2 = y2[im1] - m2*x[im1]
    xZero = (b1 - b2)/(m2 - m1)
    yZero = m1*xZero + b1
    return (xZero, yZero)

@register("FlucStruc")
def fsplot_phase(input_data, closed=True, hold=0):
    """ plot the phase of a flucstruc, optionally replicating the last point
    at the beginning (if closed=True).
    This version does not yet attempt to take into account angles, or check 
    that adjacent channels are adjacent (i.e. ch2-ch1, ch2-c2 etc).
    Channel names are taken from the fs and plotted abbreviated

    1/1/2011: TODO This appears to work only for database=None config
    1/17/2011:  bdb: May be fixed - I had used channel instead of channel.name
    """
    # extract by channels
    ch1n,ch2n,ch12n,dp = [],[],[],[]
    # bdb this line should be replaced by a call to a routine names something
    #like <plotted_width> to help in deciding if the label will fit on the 
    #current graph.
    if (2*len(input_data.dphase)*len(input_data.dphase[0].item.channel_1.name))> 50:
        sep = '\n-'
    else: sep = '-'
    #sep = '-'
    for dpn in input_data.dphase:
        ch1n.append(dpn.item.channel_1.name)
        ch2n.append(dpn.item.channel_2.name)
        ch12n.append(dpn.item.channel_1.name+sep+dpn.item.channel_2.name)
        dp.append(dpn.item.delta)

    short_names_1,p,s = split_names(ch1n)  # need to break up loops to do this
    short_names_2,p,s = split_names(ch2n)  # 

# need to know how big the shortened names are before deciding on the separator
    if (2*len(input_data.dphase)*len(short_names_1[0]))> 50:
        sep = '\n-'
    else: sep = '-'

    ch12n = [ch1n[i]+sep+ch2n[i] for i in range(len(ch1n))]
    short_ch12n = [short_names_1[i]+sep+short_names_2[i] 
                   for i in range(len(short_names_1))]

    if closed:
        ch1n.insert(0, ch1n[-1])
        ch2n.insert(0, ch2n[-1])
        ch12n.insert(0, ch12n[-1])
        short_ch12n.insert(0, short_ch12n[-1])
        dp.insert(0,dp[-1])

    pl.plot(dp,hold=hold)
    ax=pl.gca()
    ax.set_xticks(range(len(dp)))
    ax.set_xticklabels(short_ch12n)
    pl.show()

@register("SVDData")
def svdplot(input_data, fmax=None, hold=0):

    if hold==0: pl.clf(); # erase the figure, as this is a mult- axis plot

    n_SV = len(input_data.svs)

    #for chrono in input_data.chronos:
    #    print peak_freq(chrono, input_data.dim1)

    # define axes 
    ax1 = pl.subplot(221)
    ax2 = pl.subplot(222)
    ax3 = pl.subplot(223)
    ax4 = pl.subplot(224)

    # allow space for check buttons
    pl.subplots_adjust(left=0.2)

    # setup check boxes
    rax = pl.axes([0.01, 0.05, 0.09, 0.9])

    # CheckButtons take tuple arguments, tuples are immutable, so create lists fom svd info, and then cast as tuple
    button_name_list=[]
    button_setting_list=[]

    for i in range(n_SV):
	button_name_list.append('  '+str(i))
	button_setting_list.append(False)
        
    # have first 2 SVs on as default
    for i in [0,1]:
	button_setting_list[i] = True

    check = CheckButtons(rax, tuple(button_name_list), tuple(button_setting_list))
    # hack to make check buttons square
    check_box_stretch = 7
    for i in range(len(check.rectangles)):
	check.rectangles[i].set_width(check_box_stretch*check.rectangles[i].get_height())
	for j in [0,1]: # two lines of the x
            orig_x_data = check.lines[i][j].get_xdata()
            orig_y_data = check.lines[i][j].get_ydata()
            orig_width = orig_x_data[1]-orig_x_data[0]
            new_width = check_box_stretch*orig_width
            new_x_data = [orig_x_data[0],orig_x_data[0]+new_width]
            check.lines[i][j].set_data(new_x_data,orig_y_data)

    # plot all SVs, use button_setting_list for initial visibility
    # axes 1: chrono
    pl.axes(ax1)
    #pl.xlabel('Time -*-get units from Timebase-*-')
    pl.ylabel('Amplitude [a.u.]')
    plot_list_1 = range(n_SV)
    for sv_i in range(n_SV):
	#plot_list_1[sv_i], = ax1.plot(array(input_data.dim1), input_data.chronos[sv_i], visible= button_setting_list[sv_i],alpha=0.5)
	plot_list_1[sv_i], = ax1.plot(np.arange(len(input_data.chronos[sv_i])), input_data.chronos[sv_i], visible= button_setting_list[sv_i],alpha=0.5)
    #pl.xlim(min(input_data.dim1), max(input_data.dim1))

    # axes 2: SVs
    plot_list_2 = range(n_SV)
    pl.axes(ax2)
    sv_sv = [input_data.svs[i] for i in range(n_SV)]
    ax2.semilogy(np.arange(n_SV),sv_sv,'ko',markersize=3)
    entropy = input_data.H
    pl.xlabel('Singular Value number')
    pl.ylabel('Singular Value')
#    pl.figtext(0.75,0.83,'1/H = %.2f' %(1./entropy),fontsize=12, color='r')
#    pl.figtext(0.75,0.81,'H = %.2f' %(entropy),fontsize=12, color='b')
# Use kwargs so that most formatting is common to all three labels.    
    kwargs={'fontsize':12,'transform':ax2.transAxes,
            'horizontalalignment':'right'}
    
    ax2.text(0.96,0.91,'tmid = %.1fms' %(1e3*np.average(input_data.chrono_labels)), color='r', **kwargs)
    ax2.text(0.96,0.83,'1/H = %.2f' %(1./entropy), color='r', **kwargs)
    ax2.text(0.96,0.75,'H = %.2f' %(entropy), color='b', **kwargs)
    energy = Energy(input_data.p,button_setting_list)
    # this is done in two places - potential for inconsistency - wish I knew better -dgp
    # These changes make it easier to adjust the subplot layout
    # was pl.figtext(0.75,0.78, (relative to figure), make it relative to axes
    energy_label = ax2.text(0.96,0.67,'E = %.1f %%' %(100.*energy.value),
                            color='b', **kwargs)
    # grid('True')
    for sv_i in range(n_SV):
	col = plot_list_1[sv_i].get_color()
	plot_list_2[sv_i], = ax2.semilogy([sv_i], [input_data.svs[sv_i]], '%so' %(col),visible= button_setting_list[sv_i],markersize=8,alpha=0.5)

    # axes 3: fft(chrono)
    pl.axes(ax3)
    plot_list_3 = range(n_SV)
    pl.xlabel('Frequency [kHz]')
    pl.ylabel('Power Spectrum')
    pl.grid(True)            # matplotlib 1.0.X wants a boolean (unquoted)
    nyquist_kHz = 1.e-3*0.5/np.average(np.diff(input_data.chrono_labels))
    for sv_i in range(n_SV):
        col = plot_list_1[sv_i].get_color()
        tmp_chrono = input_data.chronos[sv_i]
        tmp_fft = np.fft.fft(tmp_chrono)[:len(tmp_chrono)/2]
        freq_array = nyquist_kHz*np.arange(len(tmp_fft))/(len(tmp_fft)-1)
        plot_list_3[sv_i], = ax3.plot(freq_array, abs(tmp_fft), col,visible= button_setting_list[sv_i],alpha=0.5)
        
    if fmax == None: fmax = nyquist_kHz
    pl.xlim(0,fmax)

    # axes 4: topo
    pl.axes(ax4)
    plot_list_4 = range(n_SV)
    pl.xlabel('Channel')
    pl.ylabel('Topo [a.u.]')
    angle_array = np.arange(n_SV+1)
    #channel_names = input_data.timesegment.data[input_data.diagnostic.name].ordered_channel_list
    #channel_names.append(channel_names[0])
    #pl.xticks(angle_array,channel_names, rotation=90)
    for sv_i in range(n_SV):
	col = plot_list_1[sv_i].get_color()
	tmp_topo = join_ends(input_data.topos[sv_i])
	pos,neg =  posNegFill(angle_array,np.zeros(len(angle_array)),tmp_topo)
	### BUG: it looks like ax4.fill doesn't work in a couple of cases, leaving sub_plot_4_list[i] as int, which raises a set_visible() bug in button_action - also has problems with draw(). other subplots all worked fine before I started with subplot 4
	sub_plot_4_list = range(len(pos)+len(neg)+2)
	for j in range(len(pos)):
            sub_plot_4_list[j], = ax4.fill(pos[j][0],pos[j][1],col,visible= button_setting_list[sv_i],alpha=0.5)
        for j in range(len(neg)):
            sub_plot_4_list[j+len(pos)], = ax4.fill(neg[j][0],neg[j][1],col,visible= button_setting_list[sv_i],alpha=0.5)
		
	sub_plot_4_list[len(neg)+len(pos)+0], = ax4.plot(angle_array,tmp_topo,'%so' %(col),visible= button_setting_list[sv_i],markersize=3)
	# show repeated val
	sub_plot_4_list[len(neg)+len(pos)+1], = ax4.plot([angle_array[-1]],[tmp_topo[-1]],'kx', visible= button_setting_list[sv_i],markersize=6)
	plot_list_4[sv_i]=sub_plot_4_list

    def button_action(label):
	# this is not very clear: but basically, the label is the str() of the element of plot_list_x we want to make / unmake visible
	visible_status = plot_list_1[int(label)].get_visible()
	plot_list_1[int(label)].set_visible(not visible_status)
	plot_list_2[int(label)].set_visible(not visible_status)
	plot_list_3[int(label)].set_visible(not visible_status)
	for i in range(len(plot_list_4[int(label)])):
            plot_list_4[int(label)][i].set_visible(not visible_status)
	# if visible_status == False, then we are adding visiblity => add to energy, vice-verca
	if visible_status:
            energy.sub(int(label))
	else:
            energy.add(int(label))
	energy_label._text='E = %.2f %%' %(100.*energy.value)
	pl.draw()

    # action when button is clicked
    check.on_clicked(button_action)

    # show plot
    pl.show()
    pass
