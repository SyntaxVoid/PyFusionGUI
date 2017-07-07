# GUI/main.py #
# John Gresl 6/19/2017 #

# Python
import random
import os
from collections import OrderedDict
import threading
import backend

# Anaconda
import matplotlib.pyplot as plt
import numpy as np

# PyFusion
from PyFusionGUI import *  # This contains the default file locations of import directories.
from pyfusion import DEFAULT_CONFIG_FILE

# My Additions
from Utilities import jtools as jt
from Analysis import analysis, point_analysis

# tkinter
try:
    import tkinter as tk  # Will work with python 3
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror
except ImportError:
    import Tkinter as tk  # Will work with python 2
    from tkFileDialog import askopenfilename, asksaveasfilename
    from tkMessageBox import showerror, showinfo

font_name = "Arial"
font = (font_name, 14)

DEFAULT_SETTINGS_DIR = os.path.join(GUI_DIR, ".guiconfig")


def TEST_PLOTTER():
    figure1, axes1 = plt.subplots(nrows=2, ncols=2)
    figure2, axes2 = plt.subplots(nrows=2, ncols=2)
    axesf1 = axes1.flatten()
    axesf2 = axes2.flatten()
    import random
    x = np.linspace(0,6*np.pi,100)
    for curax1, curax2 in zip(axesf1, axesf2):
        amp = random.uniform(0.2, 1.2)
        curax1.plot(amp*np.cos(x))
        curax2.plot(amp*np.sin(x))
    return ((figure1, axes1), (figure2, axes2))

class TesterClass:
    def __init__(self):
        return
    def save(self, f):
        print("Saved (not really) to", f)
    def restore(self, f):
        print("Restored (not really) from", f)


class ErrorWindow:
    def __init__(self, master, message):
        showerror(master=master, title="Error!", message=message)
        return


class ClusteringWindow:
    def __init__(self, master):
        self.root = tk.Toplevel(master=master)
        self.message_frame = tk.Frame(master=self.root)
        self.message_frame.grid(row=0, column=0, sticky=tk.N)
        self.buttons_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.buttons_frame.grid(row=1, column=0, sticky=tk.N)
        self.root.resizable(height=False, width=False)
        self.root.title("Clustering in Progress")

        self.message = tk.StringVar(master=self.message_frame, value="Now clustering.\nPlease wait.")
        self.label = tk.Label(master=self.message_frame, textvariable=self.message, font=(font_name, 24))
        self.label.grid(row=0, column=0, sticky=tk.N)
        self.root.grab_set()
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_no_close)
        self.root.bind("<<clustering_complete>>", self.clustering_complete)
        self.root.bind("<<clustering_failed>>", self.clustering_failed)

    def clustering_complete(self, e):
        # When clustering is complete, a window should pop up asking the user what they want to do.
        # Whether they want to save the actual objects, save the plots, show the plots or to close.
        size = {"height": 2, "width": 16}
        self.root.title("Clustering Complete!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        self.message.set("Clustering complete!\nPlease select an option.")
        self.root.resizable(width=False, height=False)
        object_save_button = tk.Button(master=self.buttons_frame,
                                       text="Save Analysis\nObject",
                                       font=(font_name, 18),
                                       command=self.save_objects,
                                       **size)
        object_save_button.grid(row=0, column=0, sticky=tk.N)
        plot_button = tk.Button(master=self.buttons_frame,
                                text="Plot clusters",
                                font=(font_name, 18),
                                command=self.plot_clusters,
                                **size)
        plot_button.grid(row=0, column=1, sticky=tk.N)
        close_button = tk.Button(master=self.buttons_frame,
                                 text="Close",
                                 font=(font_name, 18),
                                 command=self.root.destroy,
                                 **size)
        close_button.grid(row=0, column=2, sticky=tk.N)
        return

    def clustering_failed(self, e):
        self.root.title("Clustering Failed!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        self.message.set("Clustering Failed!")
        self.ok_button = tk.Button(master=self.buttons_frame,
                                   text="OK", font=(font_name, 18),
                                   command=self.root.destroy)
        self.ok_button.grid(row=0, column=0, sticky=tk.N)
        return

    def save_objects(self):
        fname = asksaveasfilename(initialdir=PICKLE_SAVE_DIR,
                              filetypes=(("Analysis Object File", "*.ANobj"), ("All Files", "*.*")))
        if fname == "":
            return None
        self.AN.save(fname)
        return

    def plot_clusters(self):
        plot1, plot2 = self.AN.return_plots()
        plt.show()
        return

    def x_no_close(self):
        # Uncomment this when I figure out how to cancel the analysis that is already in progress.
        #  popup = tk.Toplevel(master=self.root)
        # popup.resizable(width=False, height=False)
        # message = tk.Label(master=popup, text="Do you really wish to close?", font=font)
        # message.grid(row=0, column=0, columnspan=2, sticky=tk.N)
        # yes = tk.Button(master=popup, text="Yes", command=self.root.destroy, font=font)
        # yes.grid(row=1, column=0, sticky=tk.N)
        # no = tk.Button(master=popup, text="No", command=popup.destroy, font=font)
        # no.grid(row=1, column=1, sticky=tk.N)
        return

    def x_close(self):
        self.root.destroy()
        return

class PinpointWindow:
    def __init__(self, master, defaults, pf_window):
        self.pf_window = pf_window
        self.root = tk.Toplevel(master=master)
        self.root.geometry("410x350")
        self.root.resizable(width=False, height=False)
        self.heading_label = tk.Label(master=self.root, text="Pinpoint Analysis", font=(font_name, 24))
        self.heading_label.grid(row=0, column=0, columnspan=2, sticky=tk.N)
        self.shot_label = tk.Label(master=self.root, text="Shot:", font=font)
        self.shot_label.grid(row=1, column=0, sticky=tk.NE)
        self.shot_var = tk.StringVar(master=self.root)
        self.shot_entry = tk.Entry(master=self.root, font=font, textvariable=self.shot_var)
        self.shot_entry.grid(row=1, column=1, sticky=tk.N)
        self.shot_help_label = tk.Label(master=self.root, text="Enter shot number", font=(font_name, 9))
        self.shot_help_label.grid(row=2, column=0, columnspan=2, sticky=tk.N)
        self.time_window_label = tk.Label(master=self.root, text="Time Window (ms):", font=font)
        self.time_window_label.grid(row=3, column=0, sticky=tk.NE)
        self.time_window_var = tk.StringVar(master=self.root)
        self.time_window_entry = tk.Entry(master=self.root, font=font, textvariable=self.time_window_var)
        self.time_window_entry.grid(row=3, column=1, sticky=tk.N)
        self.time_window_help_label = tk.Label(master=self.root,
                                               text="Enter time window (plotting purposes only). e.g. 300-500",
                                               font=(font_name, 9))
        self.time_window_help_label.grid(row=4, column=0, columnspan=2, sticky=tk.N)
        self.freq_range_label = tk.Label(master=self.root, text="Freq. Range (kHz):", font=font)
        self.freq_range_label.grid(row=5, column=0, sticky=tk.NE)
        self.freq_range_var = tk.StringVar(master=self.root)
        self.freq_range_entry = tk.Entry(master=self.root, font=font, textvariable=self.freq_range_var)
        self.freq_range_entry.grid(row=5, column=1, sticky=tk.N)
        self.freq_range_help_label = tk.Label(master=self.root,
                                              text="Enter frequency range (plotting purposes only). e.g. 50-250",
                                              font=(font_name, 9))
        self.freq_range_help_label.grid(row=6, column=0, columnspan=2, sticky=tk.N)
        self.time_label = tk.Label(master=self.root, text="Time (ms):", font=font)
        self.time_label.grid(row=7, column=0, sticky=tk.NE)
        self.time_var = tk.StringVar(master=self.root)
        self.time_entry = tk.Entry(master=self.root, font=font, textvariable=self.time_var)
        self.time_entry.grid(row=7, column=1, sticky=tk.N)
        self.time_help_label = tk.Label(master=self.root,
                                        text="Enter the time you would like to pinpoint. e.g. 534.23",
                                        font=(font_name, 9))
        self.time_help_label.grid(row=8, column=0, columnspan=2, sticky=tk.N)
        self.freq_label = tk.Label(master=self.root, text="Freq. (kHz):", font=font)
        self.freq_label.grid(row=9, column=0, sticky=tk.NE)
        self.freq_var = tk.StringVar(master=self.root)
        self.freq_entry = tk.Entry(master=self.root, font=font, textvariable=self.freq_var)
        self.freq_entry.grid(row=9, column=1, sticky=tk.N)
        self.freq_help_label = tk.Label(master=self.root,
                                        text="Enter the freq you would like to pinpoint. e.g. 63.42",
                                        font=(font_name, 9))
        self.freq_help_label.grid(row=10, column=0, columnspan=2, sticky=tk.N)
        self.button_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.button_frame.grid(row=11, column=1, sticky=tk.E)
        self.ok_button = tk.Button(master=self.button_frame, text="Continue", font=font, command=self.ok)
        self.ok_button.grid(row=0, column=0, sticky=tk.E)
        self.cancel_button = tk.Button(master=self.button_frame, text="Cancel", font=font, command=self.root.destroy)
        self.cancel_button.grid(row=0, column=1, sticky=tk.W)
        self.root.grab_set()
        self.analysis_message = tk.StringVar(master=self.root)
        self.root.bind("<<analysis_complete>>", self.analysis_complete)

        if defaults is not None:
            self.shot_var.set(defaults[0])
            self.time_window_var.set(defaults[1])
            self.freq_range_var.set(defaults[2])
        return

    def analysis_complete(self, e):
        shot, time_window, freq_range, time, freq = self.get_vars()
        time_window = jt.time_window_parser(time_window)
        freq_range = jt.time_window_parser(freq_range)
        fig, \
        ax1,  \
        ax2,   \
        ax3 = self.A.return_pinpoint_plots(shot, time, freq, clusters="all")
            # point_analysis.point_analysis(A=self.A, shot=shot, time_window=time_window,
            #                                 t0=time, f0=freq,
            #                                 probe_array=self.pf_window.value_dict["probe_array"].get())
        self.analysis_message.set("Analysis complete!")
        plt.show()
        return

    def valid_values(self):
        valid = True
        try:
            int(self.shot_var.get())
        except ValueError:
            valid = False
            ErrorWindow(self.root, "Shot entry is invalid.")
        if not jt.valid_window(self.time_window_var.get()):
            valid = False
            ErrorWindow(self.root, "Time window entry is invalid.")
        if not jt.valid_window(self.freq_range_var.get()):
            valid = False
            ErrorWindow(self.root, "Freq. range entry is invalid.")
        try:
            float(self.time_var.get())
        except ValueError:
            valid = False
            ErrorWindow(self.root, "Time entry is invalid.")
        try:
            float(self.freq_var.get())
        except ValueError:
            valid = False
            ErrorWindow(self.root, "Freq. entry is invalid.")
        return valid

    def get_vars(self):
        return self.shot_var.get(), self.time_window_var.get(), \
               self.freq_range_var.get(), float(self.time_var.get()), float(self.freq_var.get())

    def ok(self):
        def callback():
            if not self.valid_values():
                return
            else:
                shot, time_window, freq_range, time, freq = self.get_vars()
                time_window = jt.time_window_parser(time_window)
                freq_range = jt.time_window_parser(freq_range)
                # TODO: What if A is None
                self.A = self.pf_window.settings_to_analysis_object()
                self.root.event_generate("<<analysis_complete>>", when="tail")
        popup = tk.Toplevel(master=self.root)
        popup.resizable(width=False, height=False)
        self.analysis_message.set("Now performing pinpoint analysis.\nPlease wait.")
        message = tk.Label(master=popup, textvariable=self.analysis_message, font=(font_name, 24))
        message.grid(row=0, column=0, sticky=tk.N)
        popup.grab_set()
        t = threading.Thread(target=callback)
        t.start()
        return

class PyFusionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(width=False, height=False)
        self.root.title("PyFusion GUI v. 0")
        self.root.geometry("1070x500")
        self.value_dict = OrderedDict()

        # ======================================
        # ======================================
        # ==         SHOT INFORMATION         ==
        # ======================================
        # ======================================
        self.shot_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.shot_frame.grid(padx=15, pady=15, row=0, column=0, sticky=tk.N + tk.W)
        self.shot_info = tk.Label(master=self.shot_frame,
                                  text="Shot Info",
                                  font=(font_name, 20))
        self.shot_info.grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)
        self.shot_label = tk.Label(master=self.shot_frame,
                                   text="Shots:",
                                   font=font)
        self.shot_label.grid(row=1, sticky=tk.E)
        self.shot_help_label = tk.Label(master=self.shot_frame,
                                        text="Enter shot numbers and/or shot ranges separated" +
                                             "\nby commas. For example, 159243,159248-159255.",
                                        font=(font_name, 9))
        self.shot_help_label.grid(row=2, column=0, columnspan=2)
        self.shot_var = tk.StringVar(master=self.shot_frame)
        self.shots = tk.Entry(master=self.shot_frame,
                              font=font,
                              textvariable=self.shot_var)
        self.shots.grid(row=1, column=1, padx=5)

        self.time_label = tk.Label(master=self.shot_frame,
                                   text="Times:",
                                   font=font)
        self.time_label.grid(row=3, sticky=tk.E)
        self.time_help_label = tk.Label(master=self.shot_frame,
                                        text="Enter a time range in ms. For example, 300-1400.",
                                        font=(font_name, 9))
        self.time_help_label.grid(row=4, column=0, columnspan=2)
        self.time_var = tk.StringVar(master=self.shot_frame)
        self.times = tk.Entry(master=self.shot_frame,
                              font=font,
                              textvariable=self.time_var)
        self.times.grid(row=3, column=1, padx=5)

        self.probes_label = tk.Label(master=self.shot_frame,
                                     text="Probe Array:",
                                     font=font)
        self.probes_label.grid(row=5, sticky=tk.E)
        self.probes_help_label = tk.Label(master=self.shot_frame,
                                          text="Enter a probe array name. For example, DIIID_toroidal_mag." +
                                          "\nThese are defined in pyfusion.cfg",
                                          font=(font_name, 9))
        self.probes_help_label.grid(row=6, column=0, columnspan=2)
        self.probe_var = tk.StringVar(master=self.shot_frame)
        self.probes = tk.Entry(master=self.shot_frame,
                               font=font,
                               textvariable=self.probe_var)
        self.probes.grid(row=5, column=1, padx=5)

        # ======================================
        # ======================================
        # ==       DATAMINING SETTINGS        ==
        # ======================================
        # ======================================
        self.dms_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.dms_frame.grid(padx=7, pady=15, row=0, column=1, sticky=tk.N + tk.E)
        self.dms_info = tk.Label(master=self.dms_frame,
                                 text="Datamining Settings",
                                 font=(font_name, 20))
        self.dms_info.grid(row=0, column=0, columnspan=4, sticky=tk.NSEW)
        self.ncpus_label = tk.Label(master=self.dms_frame,
                                    text="n_cpus:",
                                    font=font)
        self.ncpus_label.grid(row=1, column=0, sticky=tk.E)
        self.ncpus_help_label = tk.Label(master=self.dms_frame,
                                         text="Enter number of CPU's to use.",
                                         font=(font_name, 9))
        self.ncpus_help_label.grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)
        self.ncpus_var = tk.StringVar(master=self.dms_frame)
        self.ncpus_entry = tk.Entry(master=self.dms_frame,
                                    width=9,
                                    font=font,
                                    textvariable=self.ncpus_var)
        self.ncpus_entry.grid(row=1, column=1, padx=5, sticky=tk.W)
        self.nclusters_label = tk.Label(master=self.dms_frame,
                                        text="n_clusters:",
                                        font=font)
        self.nclusters_label.grid(row=3, column=0, sticky=tk.E)
        self.nclusters_help_label = tk.Label(master=self.dms_frame,
                                             text="Enter number of clusters to look for.",
                                             font=(font_name, 9))
        self.nclusters_help_label.grid(row=4, column=0, columnspan=2, sticky=tk.NSEW)
        self.nclusters_var = tk.StringVar(master=self.dms_frame)
        self.nclusters_entry = tk.Entry(master=self.dms_frame,
                                        width=9,
                                        font=font,
                                        textvariable=self.nclusters_var)
        self.nclusters_entry.grid(row=3, column=1, padx=5, sticky=tk.W)
        self.niter_label = tk.Label(master=self.dms_frame,
                                    text="n_iterations:",
                                    font=font)
        self.niter_label.grid(row=5, column=0, sticky=tk.E)
        self.niter_help_label = tk.Label(master=self.dms_frame,
                                         text="Enter number of clustering iterations.",
                                         font=(font_name, 9))
        self.niter_help_label.grid(row=6, column=0, columnspan=2, sticky=tk.NSEW)
        self.niter_var = tk.StringVar(master=self.dms_frame)
        self.niter_entry = tk.Entry(master=self.dms_frame,
                                    font=font,
                                    width=9,
                                    textvariable=self.niter_var)
        self.niter_entry.grid(row=5, column=1, padx=5, sticky=tk.W)
        self.start_method_label = tk.Label(master=self.dms_frame,
                                           text="Start Method:",
                                           font=font)
        self.start_method_label.grid(row=7, column=0, sticky=tk.E)
        self.start_method_help_label = tk.Label(master=self.dms_frame,
                                                text="Enter starting clustering method.",
                                                font=(font_name, 9))
        self.start_method_help_label.grid(row=8, column=0, columnspan=2, sticky=tk.NSEW)
        self.start_method_var = tk.StringVar(master=self.dms_frame)
        self.start_method_entry = tk.Entry(master=self.dms_frame,
                                           font=font,
                                           width=9,
                                           textvariable=self.start_method_var)
        self.start_method_entry.grid(row=7, column=1, padx=5, sticky=tk.W)
        self.method_label = tk.Label(master=self.dms_frame,
                                     text="Method:",
                                     font=font)
        self.method_label.grid(row=9, column=0, sticky=tk.E)
        self.method_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter final clustering method.",
                                          font=(font_name, 9))
        self.method_help_label.grid(row=10, column=0, columnspan=2, sticky=tk.NSEW)
        self.method_var = tk.StringVar(master=self.dms_frame)
        self.method_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.method_var)
        self.method_entry.grid(row=9, column=1, padx=5, sticky=tk.W)
        self.freq_range_label = tk.Label(master=self.dms_frame,
                                         text="Freq. range:",
                                         font=font)
        self.freq_range_label.grid(row=11, column=0, sticky=tk.E)
        self.freq_range_help_label = tk.Label(master=self.dms_frame,
                                              text="Enter frequency range in kHz.\nFor example, 50-250.",
                                              font=(font_name, 9))
        self.freq_range_help_label.grid(row=12, column=0, columnspan=2, sticky=tk.NSEW)
        self.freq_range_var = tk.StringVar(master=self.dms_frame)
        self.freq_range_entry = tk.Entry(master=self.dms_frame,
                                         font=font,
                                         width=9,
                                         textvariable=self.freq_range_var)
        self.freq_range_entry.grid(row=11, column=1, padx=5, sticky=tk.W)

        self.seed_label = tk.Label(master=self.dms_frame,
                                   text="Seed:",
                                   font=font)
        self.seed_label.grid(row=1, column=3, sticky=tk.E)
        self.seed_help_label = tk.Label(master=self.dms_frame,
                                        text="Enter clustering seed.",
                                        font=(font_name, 9))
        self.seed_help_label.grid(row=2, column=3, columnspan=2, sticky=tk.NSEW)
        self.seed_var = tk.StringVar(master=self.dms_frame)
        self.seed_entry = tk.Entry(master=self.dms_frame,
                                   font=font,
                                   width=9,
                                   textvariable=self.seed_var)
        self.seed_entry.grid(row=1, column=4, padx=5, sticky=tk.W)
        self.random_seed_button = tk.Button(master=self.dms_frame,
                                            text="Random Seed",
                                            font=(font_name, 9),
                                            command=self.random_seed,
                                            bd=3)
        self.random_seed_button.grid(row=3, column=4, sticky=tk.NSEW)

        self.npeaks_label = tk.Label(master=self.dms_frame,
                                     text="n_peaks:",
                                     font=font).grid(row=5, column=3, sticky=tk.E)
        self.npeaks_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter number of peaks to find.",
                                          font=(font_name, 9))
        self.npeaks_help_label.grid(row=6, column=3, columnspan=2, sticky=tk.NSEW)
        self.npeaks_var = tk.StringVar(master=self.dms_frame)
        self.npeaks_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.npeaks_var)
        self.npeaks_entry.grid(row=5, column=4, padx=5, sticky=tk.W)

        self.cutoff_label = tk.Label(master=self.dms_frame,
                                     text="Cutoff By:",
                                     font=font)
        self.cutoff_label.grid(row=7, column=3, sticky=tk.E)
        self.cutoff_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter cutoff by item.",
                                          font=(font_name, 9))
        self.cutoff_help_label.grid(row=8, column=3, columnspan=2, sticky=tk.NSEW)
        self.cutoff_var = tk.StringVar(master=self.dms_frame)
        self.cutoff_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.cutoff_var)
        self.cutoff_entry.grid(row=7, column=4, padx=5, sticky=tk.W)
        self.cutoff_val_label = tk.Label(master=self.dms_frame,
                                         text="Cutoff Value:",
                                         font=font)
        self.cutoff_val_label.grid(row=9, column=3, sticky=tk.E)
        self.cutoff_val_help_label = tk.Label(master=self.dms_frame,
                                              text="Enter filter cutoff value.",
                                              font=(font_name, 9))
        self.cutoff_val_help_label.grid(row=10, column=3, columnspan=2, sticky=tk.NSEW)
        self.cutoff_val_var = tk.StringVar(master=self.dms_frame)
        self.cutoff_val_entry = tk.Entry(master=self.dms_frame,
                                         font=font,
                                         width=9,
                                         textvariable=self.cutoff_val_var)
        self.cutoff_val_entry.grid(row=9, column=4, padx=5, sticky=tk.W)
        self.filter_item_label = tk.Label(master=self.dms_frame,
                                          text="Filter Items:",
                                          font=font)
        self.filter_item_label.grid(row=11, column=3, sticky=tk.E)
        self.filter_item_help_label = tk.Label(master=self.dms_frame,
                                               text="Enter filter items.",
                                               font=(font_name, 9))
        self.filter_item_help_label.grid(row=12, column=3, columnspan=2, sticky=tk.NSEW)
        self.filter_item_var = tk.StringVar(master=self.dms_frame)
        self.filter_item_entry = tk.Entry(master=self.dms_frame,
                                          font=font,
                                          width=9,
                                          textvariable=self.filter_item_var)
        self.filter_item_entry.grid(row=11, column=4, padx=5, sticky=tk.W)

        # ======================================
        # ======================================
        # ==        SETTINGS BUTTONS          ==
        # ======================================
        # ======================================
        self.button_frame = tk.Frame(master=self.root)
        self.button_frame.grid(row=0, column=2, sticky=tk.N)
        self.settings_buttons_frame = tk.Frame(master=self.button_frame, bd=5, relief=tk.SUNKEN)
        self.settings_buttons_frame.grid(row=0, column=0, sticky=tk.N, padx=15, pady=15)

        self.settings_button_heading = tk.Label(master=self.settings_buttons_frame,
                                                text="Settings Options",
                                                font=font)
        self.settings_button_heading.grid(row=0, column=0, sticky=tk.N)
        self.save_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Save Settings",
                                              font=(font_name, 13), width=14,
                                              command=self.save_settings)
        self.save_settings_button.grid(row=1, column=0, sticky=tk.N)
        self.load_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Load From File",
                                              font=(font_name, 13), width=14,
                                              command=self.load_settings)
        self.load_settings_button.grid(row=2, column=0, sticky=tk.N)
        self.restore_defaults_button = tk.Button(master=self.settings_buttons_frame,
                                                 text="Restore Defaults",
                                                 font=(font_name, 13), width=14,
                                                 command=self.restore_defaults)
        self.restore_defaults_button.grid(row=3, column=0, sticky=tk.N)

        # ======================================
        # ======================================
        # ==        CLUSTERING BUTTONS        ==
        # ======================================
        # ======================================
        self.analysis_frame = tk.Frame(master=self.button_frame, bd=5, relief=tk.SUNKEN)
        self.analysis_frame.grid(row=1, column=0, sticky=tk.N, padx=15, pady=15)
        self.analysis_heading = tk.Label(master=self.analysis_frame,
                                         text="Analysis Options",
                                         font=font)
        self.analysis_heading.grid(row=0, column=0, sticky=tk.N)
        self.run_clustering_button = tk.Button(master=self.analysis_frame,
                                               text="Cluster from\nCurrent Settings",
                                               font=(font_name, 13), width=14,
                                               command=self.run_clustering)
        self.run_clustering_button.grid(row=1, column=0, sticky=tk.N)
        self.restore_clustering_button = tk.Button(master=self.analysis_frame,
                                                   text="Restore from\nPrevious Analysis",
                                                   font=(font_name, 13), width=14,
                                                   command=self.restore_clustering)
        self.restore_clustering_button.grid(row=2, column=0, sticky=tk.N)
        self.run_point_analysis_button = tk.Button(master=self.analysis_frame,
                                                   text="Pinpoint Analysis",
                                                   font=(font_name, 13), width=14,
                                                   command=self.run_point_analysis)
        self.run_point_analysis_button.grid(row=3, column=0, sticky=tk.N)
        self.run_point_analysis_help = tk.Label(master=self.analysis_frame,
                                                text="For plotting data at a certain"
                                                     "\ntime and frequency.",
                                                font=(font_name, 9))
        self.run_point_analysis_help.grid(row=4, column=0, sticky=tk.N)
        # ======================================
        # ======================================
        # ==          MISC. WIDGETS           ==
        # ======================================
        # ======================================
        self.misc_frame = tk.Frame(master=self.button_frame, bd=5, relief=tk.SUNKEN)
        self.misc_frame.grid(row=2, column=0, sticky=tk.N)
        self.misc_heading = tk.Label(master=self.misc_frame,
                                     text="Other Options",
                                     font=font)
        self.misc_heading.grid(row=0, column=0, sticky=tk.N)
        self.close_button = tk.Button(master=self.misc_frame,
                                      text="Close Window",
                                      font=(font_name, 13), width=14,
                                      command=self.root.destroy)
        self.close_button.grid(row=1, column=0, sticky=tk.N)

        # ======================================
        # ======================================
        # ==      SAVING INITIAL VALUES       ==
        # ======================================
        # ======================================
        self.value_dict["shots"] = self.shot_var
        self.value_dict["times"] = self.time_var
        self.value_dict["probe_array"] = self.probe_var
        self.value_dict["n_cpus"] = self.ncpus_var
        self.value_dict["n_clusters"] = self.nclusters_var
        self.value_dict["n_iterations"] = self.niter_var
        self.value_dict["start"] = self.start_method_var
        self.value_dict["method"] = self.method_var
        self.value_dict["freq_range"] = self.freq_range_var
        self.value_dict["seed"] = self.seed_var
        self.value_dict["n_peaks"] = self.npeaks_var
        self.value_dict["cutoff_by"] = self.cutoff_var
        self.value_dict["cutoff_value"] = self.cutoff_val_var
        self.value_dict["filter_items"] = self.filter_item_var

        return

    def start(self):
        self.restore_defaults()
        self.root.mainloop()
        return None

    def random_seed(self):
        self.seed_var.set(str(random.choice(range(200, 1000))))
        return None

    def restore_clustering(self):
        fname = askopenfilename(initialdir=GUI_DIR,
                                filetypes=(("Analysis File Object", "*.ANobj"), ("All Files", "*.*")))
        if fname == "":
            return None
        try:
            AN = analysis.Analysis.restore(fname)
            print(AN)
        except:
            ErrorWindow(self.root, "Incorrect file format.")
        return None

    def update_values(self):
        self.shot_var.set(self.value_dict["shots"])
        self.time_var.set(self.value_dict["times"])
        self.probe_var.set(self.value_dict["probe_array"])
        self.ncpus_var.set(self.value_dict["n_cpus"])
        self.nclusters_var.set(self.value_dict["n_clusters"])
        self.niter_var.set(self.value_dict["n_iterations"])
        self.start_method_var.set(self.value_dict["start"])
        self.method_var.set(self.value_dict["method"])
        self.freq_range_var.set(self.value_dict["freq_range"])
        self.seed_var.set(self.value_dict["seed"])
        self.npeaks_var.set(self.value_dict["n_peaks"])
        self.cutoff_var.set(self.value_dict["cutoff_by"])
        self.cutoff_val_var.set(self.value_dict["cutoff_value"])
        self.filter_item_var.set(self.value_dict["filter_items"])
        return None

    def valid_values(self):
        valid = True
        if not jt.valid_shot_str(self.value_dict["shots"].get()):
            ErrorWindow(self.root, "Shots entry is invalid.")
            valid = False
        if not jt.valid_window(self.value_dict["times"].get()):
            ErrorWindow(self.root, "Time window entry is invalid.")
            valid = False
        if not jt.valid_probe_array(self.value_dict["probe_array"].get()):
            ErrorWindow(self.root, "Probe array is not valid! Please enter a valid" +
                        " probe array name from:\n{}".format(jt.scan_config(DEFAULT_CONFIG_FILE)))
            valid = False
        if not jt.valid_int_from_str(self.value_dict["n_cpus"].get()):
            ErrorWindow(self.root, "Number of CPUs must be an integer.")
            valid = False
        if not jt.valid_int_from_str(self.value_dict["n_clusters"].get()):
            ErrorWindow(self.root, "Number of clusters must be an integer.")
            valid = False
        if not jt.valid_int_from_str(self.value_dict["n_iterations"].get()):
            ErrorWindow(self.root, "Number of iterations must be an integer.")
            valid = False
        if not jt.valid_method(self.value_dict["method"].get()):
            ErrorWindow(self.root, "Method must be in: {}. To add a new method to this list," +
                        " edit ""return_methods()"" in Utilities/jtools.py")
            valid = False
        if not jt.valid_window(self.value_dict["freq_range"].get()):
            ErrorWindow(self.root, "Frequency range entry is invalid.")
            valid = False
        if not jt.valid_int_from_str(self.value_dict["seed"].get()):
            ErrorWindow(self.root, "Seed must be an integer.")
            valid = False
        if not jt.valid_int_from_str(self.value_dict["n_peaks"].get()):
            ErrorWindow(self.root, "Number of peaks must be an integer.")
            valid = False
        # Not sure how to test cutoff_by...
        if not jt.valid_int_from_str(self.value_dict["cutoff_value"].get()):
            ErrorWindow(self.root, "Cutoff value must be an integer.")
            valid = False
        # Not sure how to test filter_items... :(
        # Checks the contents of "every" cell to ensure the contents are valid.
        return valid

    def load_values_from_str(self, s):
        lines = s.split("\n")
        for line in lines:
            val = line.split(":")[1].strip()
            if line.startswith("shots"):
                self.shot_var.set(val)
            elif line.startswith("times"):
                self.time_var.set(val)
            elif line.startswith("probe_array"):
                self.probe_var.set(val)
            elif line.startswith("n_cpus"):
                self.ncpus_var.set(val)
            elif line.startswith("n_clusters"):
                self.nclusters_var.set(val)
            elif line.startswith("n_iterations"):
                self.niter_var.set(val)
            elif line.startswith("start"):
                self.start_method_var.set(val)
            elif line.startswith("method"):
                self.method_var.set(val)
            elif line.startswith("freq_range"):
                self.freq_range_var.set(val)
            elif line.startswith("seed"):
                self.seed_var.set(val)
            elif line.startswith("n_peaks"):
                self.npeaks_var.set(val)
            elif line.startswith("cutoff_by"):
                self.cutoff_var.set(val)
            elif line.startswith("cutoff_value"):
                self.cutoff_val_var.set(val)
            elif line.startswith("filter_item"):
                self.filter_item_var.set(val)
        return

    def load_values_from_file(self, f):
        with open(f) as vals:
            settings = vals.read()
        self.load_values_from_str(settings)
        return

    def load_settings(self):
        fname = askopenfilename(initialdir=GUI_DIR,
                                filetypes=(("GUI Config File", "*.guiconfig"), ("All Files", "*.*")))
        if fname == "":
            return None
        try:
            self.load_values_from_file(fname)
        except:
            ErrorWindow(self.root, "Incorrect file format.")
        return None

    def save_settings(self):
        if self.valid_values():
            fname = asksaveasfilename(initialdir=GUI_DIR,
                                      filetypes=(("GUI Config File", "*.guiconfig"), ("All Files", "*.*")))
            if fname == "":
                return None
            backend.save_values(self.value_dict, fname)
            return None

    def defaults_missing(self):
        defaults = '''shots: 159243
times: 300-1400
probe_array: DIIID_toroidal_mag
n_cpus: 1
n_clusters: 16
n_iterations: 20
start: k_means
method: EM_VMM
freq_range: 50-250
seed: 743
n_peaks: 20
cutoff_by: sigma_eq
cutoff_value: 25
filter_items: EM_VMM_kappas'''
        self.load_values_from_str(defaults)
        with open(os.path.join(GUI_DIR,".guiconfig"), "w") as new_default:
            new_default.write(defaults)
        return

    def restore_defaults(self):
        try:
            self.load_values_from_file(DEFAULT_SETTINGS_DIR)
        except:
            self.defaults_missing()
        return None

    def run_clustering(self):

        def callback():
            AN = self.settings_to_analysis_object()
            win.AN = AN
            if AN is None:
                win.root.event_generate("<<clustering_failed>>", when="tail")
            else:
                win.root.event_generate("<<clustering_complete>>", when="tail")
            # import time
            # time.sleep(5)
            return

        if self.valid_values():
            win = ClusteringWindow(master=self.root)
            t = threading.Thread(target=callback)
            t.start()
        return None

    def run_point_analysis(self):
        # A window needs to popup asking the user for a frequency and time.
        # Verify both frequency and time are within our analysis windows.
        # Can only be done after the analysis has been performed. (Grey out button before that??)
        # but we'll ignore the latter for now.
        if self.valid_values():
            defaults = [str(jt.shot_str_parser(self.value_dict["shots"].get())[0]),
                        self.value_dict["times"].get(),
                        self.value_dict["freq_range"].get()]
            win = PinpointWindow(master=self.root, defaults=defaults, pf_window=self)

        return

    def settings_to_analysis_object(self):
        # Takes the current settings and creates an Analysis object from Analysis/analysis.py
        if self.valid_values():
            shots = jt.shot_str_parser(self.value_dict["shots"].get())
            time_windows = jt.time_window_parser(self.value_dict["times"].get())
            print(time_windows)
            probes = self.value_dict["probe_array"].get()
            n_cpus = int(self.value_dict["n_cpus"].get())
            n_clusters = int(self.value_dict["n_clusters"].get())
            n_iterations = int(self.value_dict["n_iterations"].get())
            start = self.value_dict["start"].get()
            method = self.value_dict["method"].get()
            freq_range = jt.time_window_parser(self.value_dict["freq_range"].get())
            lower_freq = freq_range[0]
            upper_freq = freq_range[1]
            seeds = int(self.value_dict["seed"].get())
            n_peaks = int(self.value_dict["n_peaks"].get())
            cutoff_by = self.value_dict["cutoff_by"].get()
            cutoff_value = int(self.value_dict["cutoff_value"].get())
            filter_items = self.value_dict["filter_items"].get()
            datamining_settings = {'n_clusters': n_clusters, 'n_iterations': n_iterations,
                                   'start': start, 'verbose': 0, 'method': method, "seeds": seeds}
            fft_settings = {"n_pts": n_peaks, "lower_freq": lower_freq, "upper_freq": upper_freq,
                            "cutoff_by": cutoff_by, "ave_kappa_cutoff": cutoff_value, "filter_item": filter_items}
            DM = analysis.DataMining(shots=shots, time_windows=time_windows, probes=probes,
                                     datamining_settings=datamining_settings, fft_settings=fft_settings,
                                     n_cpus=n_cpus)
            AN = analysis.Analysis(DM=DM)
            if AN.results is None or AN.feature_object is None or AN.z is None:
                ErrorWindow(master=self.root, message="No clusters found! (Maybe try increasing cutoff value...)")
                return None
            return AN
        return None

if __name__ == '__main__':
    window1 = PyFusionWindow()
    window1.start()
    # import time
    # root = tk.Tk()
    # def snoot(s):
    #     print(s)
    #     time.sleep(3)
    #     print(s)
    #     return
    #
    # def boop():
    #     vars = P.get_vars()
    #     snoot(vars)
    #     return
    #
    #
    # #t = threading.Thread(target=snoot)
    # P = PinpointWindow(root, boop)
    # root.mainloop()
    pass
