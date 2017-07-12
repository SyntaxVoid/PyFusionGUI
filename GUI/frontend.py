# GUI/main.py #
# John Gresl 6/19/2017 #

# Python
import random
import os
from collections import OrderedDict
import threading
from datetime import datetime

# Anaconda
import matplotlib.pyplot as plt
import numpy as np

# PyFusion
from PyFusionGUI import *  # This contains the default file locations of import directories.
from pyfusion import DEFAULT_CONFIG_FILE

# My Additions
from Utilities import jtools as jt
from Analysis import analysis
import backend

# tkinter
try: # Will work with python 3
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror
except ImportError: # Will work with python 2
    import Tkinter as tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    from tkMessageBox import showerror, showinfo

font_name = "Arial"
font = (font_name, 14)

DEFAULT_SETTINGS_DIR = os.path.join(GUI_DIR, ".guiconfig")


class ErrorWindow:
    def __init__(self, master, message):
        showerror(master=master, title="Error!", message=message)
        return


class ClusteringWindow:
    def __init__(self, master, slurm_start_time=None, ANobj_restore=None):
        self.root = tk.Toplevel(master=master)
        self.message_frame = tk.Frame(master=self.root)
        self.message_frame.grid(row=0, column=0, sticky=tk.N)
        self.buttons_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.buttons_frame.grid(row=1, column=0, sticky=tk.N)
        self.root.resizable(height=False, width=False)
        self.message = tk.StringVar(master=self.message_frame)
        self.label = tk.Label(master=self.message_frame, textvariable=self.message, font=(font_name, 24))
        self.label.grid(row=0, column=0, sticky=tk.N)
        self.root.grab_set()
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_no_close)
        self.root.bind("<<clustering_complete>>", self.clustering_complete)
        self.root.bind("<<clustering_failed>>", self.clustering_failed)
        self.root.bind("<<slurm_clustering_complete>>", self.slurm_clustering_complete)
        if ANobj_restore is None:
            self.default_wait_time = 5  # Wait 5 seconds before checking status
            self.AN = None
            self.slurm_start_time = slurm_start_time
            self.root.title("Clustering in Progress")
            if self.slurm_start_time is None:
                self.message.set("Now clustering.\nPlease wait.")
            if slurm_start_time is not None:
                self.slurm_done_file = IRIS_CSCRATCH_DIR+self.slurm_start_time+".slurmdone"
                self.ANobj_file = IRIS_CSCRATCH_DIR+self.slurm_start_time+".ANobj"
                self._cur = self.default_wait_time
                self.message.set("Waiting for worker\nnode to complete.\nChecking again in\n{} seconds."\
                                 .format(self._cur))
                self.root.after(1000, self.countdown)
        else:
            self.AN = ANobj_restore
            self.root.title("Analysis Object Restored")
            self.message.set("Analysis object has been restored.\nSelect an option to continue.")
            self.root.event_generate("<<clustering_complete>>", when="tail")

    def countdown(self):
        self._cur -= 1
        if self._cur <= 0:
            if self.slurm_active():
                self._cur = self.default_wait_time
            else:
                self.root.event_generate("<<slurm_clustering_complete>>", when="tail")
                return
        self.message.set("Waiting for worker\nnode to complete.\nChecking again in\n{} seconds".format(self._cur))
        self.root.after(1000, self.countdown)
        return

    def slurm_active(self):
        return not os.path.isfile(self.slurm_done_file)

    def slurm_clustering_complete(self, e):
        self.root.title("SLURM Clustering Complete!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        self.message.set("SLURM clustering complete!\nYou can now load your\nAnalysis object file from\n{}"\
                         .format(jt.break_path(self.ANobj_file, 23)))
        self.ok_button = tk.Button(master=self.root, text="OK", command=self.root.destroy, font=(font_name, 13))
        self.ok_button.grid(row=1, column=0)
        return

    def clustering_complete(self, e):
        # When clustering is complete, a window should pop up asking the user what they want to do.
        # Whether they want to save the actual objects, save the plots, show the plots or to close.
        size = {"height": 2, "width": 16}
        self.root.title("Analysis object loaded!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        self.message.set("Analysis object is loaded!\nPlease select an option.")

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
        self.label.config(fg="red")
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
        self.AN.return_specgrams()
        plt.show()
        return

    @staticmethod
    def x_no_close():
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
    def __init__(self, master, defaults, pf_window, previous_analysis=None):
        self.previous_analysis = previous_analysis
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
        self.root.bind("<<analysis_complete>>", self.analysis_complete)
        self.root.bind("<<analysis_failed>>", self.analysis_failed)

        self.popup = tk.Toplevel(master=self.root)
        self.popup.resizable(width=False, height=False)
        self.analysis_message = tk.StringVar(master=self.root, value="Now performing pinpoint analysis.\nPlease wait.")
        self.message = tk.Label(master=self.popup, textvariable=self.analysis_message, font=(font_name, 24))
        self.message.grid(row=0, column=0, sticky=tk.N)
        self.popup.withdraw()

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
        ax3 = self.AN.return_pinpoint_plots(shot=shot, t0=time, f0=freq, time_window=time_window,
                                           frequency_window=freq_range, clusters="all")
            # point_analysis.point_analysis(A=self.A, shot=shot, time_window=time_window,
            #                                 t0=time, f0=freq,
            #                                 probe_array=self.pf_window.value_dict["probe_array"].get())
        self.root.title("Analysis complete!")
        self.analysis_message.set("Analysis complete!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        ok = tk.Button(master=self.popup, text="OK", command = self.root.destroy, font=(font_name, 24))
        ok.grid(row=1, column=0, sticky=tk.N)
        plt.show()
        return


    def analysis_failed(self, e):
        self.root.title("Analysis Failed!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.x_close)
        self.analysis_message.set("Clustering Failed!")
        self.ok_button = tk.Button(master=self.button_frame, text="OK",
                                   font=(font_name, 18), command = self.root.destroy)
        self.ok_button.grid(row=0, column=0, sticky=tk.N)
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
                # shot, time_window, freq_range, time, freq = self.get_vars()
                self.popup.deiconify()
                self.popup.grab_set()
                self.popup.wm_protocol("WM_DELETE_WINDOW", self.x_no_close)
                if self.previous_analysis is None:
                    self.AN = self.pf_window.settings_to_analysis_object()
                else:
                    self.AN = self.previous_analysis
                if self.AN is None:
                    self.root.event_generate("<<analysis_failed>>", when="tail")
                else:
                    self.root.event_generate("<<analysis_complete>>", when="tail")
        # popup = tk.Toplevel(master=self.root)
        # popup.resizable(width=False, height=False)
        # self.analysis_message.set("Now performing pinpoint analysis.\nPlease wait.")
        # message = tk.Label(master=popup, textvariable=self.analysis_message, font=(font_name, 24))
        # message.grid(row=0, column=0, sticky=tk.N)
        # popup.grab_set()
        t = threading.Thread(target=callback)
        t.start()
        return

    def x_no_close(self):
        return

    def x_close(self):
        self.root.destroy()
        return


class PyFusionWindow:
    def __init__(self):
        self.AN = None
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
        self.big_frame = tk.Frame(master=self.root)
        self.big_frame.grid(row=0, column=0, sticky=tk.N)
        self.shot_frame = tk.Frame(master=self.big_frame, bd=5, relief=tk.SUNKEN)
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
        self.col_1_frame = tk.Frame(master=self.root)
        self.col_1_frame.grid(row=0, column=1, sticky=tk.N)
        self.dms_frame = tk.Frame(master=self.col_1_frame, bd=5, relief=tk.SUNKEN)
        self.dms_frame.grid(row=0, column=0)
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

        self.use_worker_node_val = tk.IntVar(master=self.root, value=1)
        self.use_worker_node_checkbox = tk.Checkbutton(master=self.col_1_frame,
                                                       text="Use Iris\nWorker Node", font=(font_name, 13),
                                                       variable=self.use_worker_node_val, bd=5, relief=tk.SUNKEN)
        self.use_worker_node_checkbox.config(state=tk.DISABLED)
        self.use_worker_node_checkbox.grid(row=1, column=0, sticky=tk.NE)

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

        self.restore_clustering_button = tk.Button(master=self.analysis_frame,
                                                   text="Restore from\nPrevious Analysis",
                                                   font=(font_name, 13), width=14,
                                                   command=self.restore_clustering)
        self.restore_clustering_button.grid(row=1, column=0, sticky=tk.N)

        self.run_clustering_button = tk.Button(master=self.analysis_frame,
                                               text="Cluster from\nCurrent Settings",
                                               font=(font_name, 13), width=14,
                                               command=self.run_clustering)
        self.run_clustering_button.grid(row=2, column=0, sticky=tk.N)

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

        self.using_analysis_var = tk.StringVar(master=self.root, value="No analysis object loaded yet.\n"\
                                               "Please perform clustering or\nrestore an anlysis object.")
        self.using_analysis_label = tk.Label(master=self.big_frame,
                                             textvariable=self.using_analysis_var,
                                             font=(font_name, 14, "bold"), fg="red")
        self.using_analysis_label.grid(row=1, column=0, sticky=tk.N)

        # ====================================== #
        # ====================================== #
        # ==             BINDINGS             == #
        # ====================================== #
        # ====================================== #
        self.root.bind("<<clustering_failed>>", self.clustering_failed)
        self.root.bind("<<clustering_complete>>", self.clustering_complete)
        self.root.bind("<<clustering_in_progress>>", self.clustering_in_progress)
        self.root.bind("<<clustering_restored>>", self.clustering_restored)

        # ====================================== #
        # ====================================== #
        # ==      SAVING INITIAL VALUES       == #
        # ====================================== #
        # ====================================== #
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

    def clustering_restored(self, e):
        win = ClusteringWindow(master=self.root, ANobj_restore=self.AN)
        return

    def clustering_failed(self, e):
        self.using_analysis_var.set("Clustering Failed. No analysis object\n"
                                    "loaded yet. Please perform clustering\n"
                                    "or restore a saved analysis object")
        self.using_analysis_label.config(fg="red")
        return

    def clustering_complete(self, e):
        self.using_analysis_var.set("Using analysis object created\n"
                                    "from custom user settings.")
        self.using_analysis_label.config(fg="dark green")
        return

    def clustering_in_progress(self, e):
        self.using_analysis_var.set("Now clustering. Please wait.")
        self.using_analysis_label.config(fg="DarkOrange1")

    def start(self):
        self.restore_defaults()
        self.root.mainloop()
        return None

    def random_seed(self):
        self.seed_var.set(str(random.choice(range(1, 10000))))
        return None

    def restore_clustering(self):
        fname = askopenfilename(initialdir=IRIS_CSCRATCH_DIR,
                                filetypes=(("Analysis File Object", "*.ANobj"), ("All Files", "*.*")))
        if fname == "" or fname == ():
            return None
        try:
            self.AN = analysis.Analysis.restore(fname)
            self._restore_settings_from_loaded_object()
            self.using_analysis_var.set("Using analysis object from\n{}".format(jt.break_path(fname, 24)))
            self.using_analysis_label.config(fg="dark green")
            win = ClusteringWindow(master=self.root, ANobj_restore=self.AN)
        except:
            ErrorWindow(self.root, "Incorrect file format.")
        return None

    def return_restored_object_values(self):
        return {"shots": jt.reverse_shot_str_parser(self.AN.DM.shot_info["shots"]),
                "times": jt.ANobj_times_to_time_window(self.AN.DM.shot_info["time_windows"]),
                "probe_array": self.AN.DM.shot_info["probes"],
                "n_cpus": str(self.AN.DM.n_cpus),
                "n_clusters": str(self.AN.DM.datamining_settings["n_clusters"]),
                "n_iterations": str(self.AN.DM.datamining_settings["n_iterations"]),
                "start": self.AN.DM.datamining_settings["start"],
                "method": self.AN.DM.datamining_settings["method"],
                "freq_range": str(self.AN.DM.fft_settings["lower_freq"])+"-"+
                                          str(self.AN.DM.fft_settings["upper_freq"]),
                "seed": str(self.AN.DM.datamining_settings["seeds"]),
                "n_peaks": str(self.AN.DM.fft_settings["n_pts"]),
                "cutoff_by": self.AN.DM.fft_settings["cutoff_by"],
                "cutoff_value": str(self.AN.DM.fft_settings["ave_kappa_cutoff"]),
                "filter_items": self.AN.DM.fft_settings["filter_item"]}

    def _restore_settings_from_loaded_object(self):
        # Loads the settings used to analyze an Analysis object that has been loaded.
        self.value_dict["shots"].set(jt.reverse_shot_str_parser(self.AN.DM.shot_info["shots"]))
        self.value_dict["times"].set(jt.ANobj_times_to_time_window(self.AN.DM.shot_info["time_windows"]))
        self.value_dict["probe_array"].set(self.AN.DM.shot_info["probes"])
        self.value_dict["n_cpus"].set(str(self.AN.DM.n_cpus))
        self.value_dict["n_clusters"].set(str(self.AN.DM.datamining_settings["n_clusters"]))
        self.value_dict["n_iterations"].set(str(self.AN.DM.datamining_settings["n_iterations"]))
        self.value_dict["start"].set(self.AN.DM.datamining_settings["start"])
        self.value_dict["method"].set(self.AN.DM.datamining_settings["method"])
        self.value_dict["freq_range"].set(str(self.AN.DM.fft_settings["lower_freq"])+"-"+
                                          str(self.AN.DM.fft_settings["upper_freq"]))
        self.value_dict["seed"].set(str(self.AN.DM.datamining_settings["seeds"]))
        self.value_dict["n_peaks"].set(str(self.AN.DM.fft_settings["n_pts"]))
        self.value_dict["cutoff_by"].set(self.AN.DM.fft_settings["cutoff_by"])
        self.value_dict["cutoff_value"].set(str(self.AN.DM.fft_settings["ave_kappa_cutoff"]))
        self.value_dict["filter_items"].set(self.AN.DM.fft_settings["filter_item"])
        return

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
            if line.startswith("shots"): self.shot_var.set(val)
            elif line.startswith("times"): self.time_var.set(val)
            elif line.startswith("probe_array"): self.probe_var.set(val)
            elif line.startswith("n_cpus"): self.ncpus_var.set(val)
            elif line.startswith("n_clusters"): self.nclusters_var.set(val)
            elif line.startswith("n_iterations"): self.niter_var.set(val)
            elif line.startswith("start"): self.start_method_var.set(val)
            elif line.startswith("method"): self.method_var.set(val)
            elif line.startswith("freq_range"): self.freq_range_var.set(val)
            elif line.startswith("seed"): self.seed_var.set(val)
            elif line.startswith("n_peaks"): self.npeaks_var.set(val)
            elif line.startswith("cutoff_by"): self.cutoff_var.set(val)
            elif line.startswith("cutoff_value"): self.cutoff_val_var.set(val)
            elif line.startswith("filter_item"): self.filter_item_var.set(val)
        return

    def load_values_from_file(self, f):
        with open(f) as vals:
            settings = vals.read()
        self.load_values_from_str(settings)
        return

    def load_settings(self):
        fname = askopenfilename(initialdir=GUI_DIR,
                                filetypes=(("GUI Config File", "*.guiconfig"), ("All Files", "*.*")))
        if fname == "" or fname == ():
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
        if self.valid_values():
            if not self.use_worker_node_val:
                def callback():
                    try:
                        self.AN = self.settings_to_analysis_object()
                        win.AN = self.AN
                        win.root.event_generate("<<clustering_complete>>", when="tail")
                        self.root.event_generate("<<clustering_complete>>", when="tail")
                    except:
                        win.root.event_generate("<<clustering_failed>>", when="tail")
                        self.root.event_generate("<<clustering_failed>>", when="tail")
                    return

                self.root.event_generate("<<clustering_in_progress>>", when="tail")
                win = ClusteringWindow(master=self.root)
                t = threading.Thread(target=callback)
                t.start()
                return None
            if self.use_worker_node_val:
                # Need to create a job script and a python script for SLURM.
                now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                pythonscript = '''from PyFusionGUI.Analysis.analysis import *
from PyFusionGUI.Utilities.jtools import *
A1 = {ANALYSIS_OBJECT}
A1.save(\"{ANOBJ_FILE}\")
write_finished_file(\"{DONE_FILE}\")
'''.format(ANALYSIS_OBJECT=self.settings_to_analysis_object_str(),
           ANOBJ_FILE=IRIS_CSCRATCH_DIR+now+".ANobj",
           DONE_FILE=IRIS_CSCRATCH_DIR+now+".slurmdone")
                with open("SLURM/temp.py", "w") as test:
                    test.write(pythonscript)
                sbatchscript = '''#!/bin/bash
#SBATCH -p short
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 5
#SBATCH --mem-per-cpu=4G
#SBATCH -o PyFusionGUI-%j.out
#SBATCH --export=ALL
#SBATCH -error ERROR.error
echo "Starting job on worker node"
/fusion/usc/opt/python/2.7.11/bin/python2.7 SLURM/temp.py
'''
                with open("SLURM/sbatch_clustering.sbatch", "w") as sbatch:
                    sbatch.write(sbatchscript)
                os.system("sbatch SLURM/sbatch_clustering.sbatch")
                win = ClusteringWindow(master=self.root, slurm_start_time=now)



    def run_point_analysis(self):
        # A window needs to popup asking the user for a frequency and time.
        # Verify both frequency and time are within our analysis windows.
        # Can only be done after the analysis has been performed. (Grey out button before that??)
        # but we'll ignore the latter for now.
        if self.AN is None:
            if self.valid_values():
                defaults = [str(jt.shot_str_parser(self.value_dict["shots"].get())[0]),
                            self.value_dict["times"].get(),
                            self.value_dict["freq_range"].get()]
            else:
                return
        else:
            object_settings = self.return_restored_object_values()
            defaults = [object_settings["shots"].split(",")[0], object_settings["times"], object_settings["freq_range"]]
        PinpointWindow(master=self.root, defaults=defaults, pf_window=self, previous_analysis=self.AN)
        return


    def settings_to_analysis_object_str(self):
        DMstr = self.settings_to_datamining_object_str()
        return self.datamining_str_to_analysis_object_str(DMstr)

    @staticmethod
    def datamining_str_to_analysis_object_str(DM):
        "Analysis(DM={})".format(DM)
        return "Analysis(DM={})".format(DM)

    def settings_to_datamining_object_str(self):
        if self.valid_values():
            shots = jt.shot_str_parser(self.value_dict["shots"].get())
            time_windows = jt.time_window_parser(self.value_dict["times"].get())
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
            DM = "DataMining(shots={shots}, time_windows={time_windows}, probes=\"{probes}\","\
                                     "datamining_settings={datamining_settings}, fft_settings={fft_settings},"\
                                     "n_cpus={n_cpus})".format(shots=shots, time_windows=time_windows, probes=probes,
                                                               datamining_settings=datamining_settings,
                                                               fft_settings=fft_settings,n_cpus=n_cpus)
            return DM

    def settings_to_analysis_object(self):
        # Takes the current settings and creates an Analysis object from Analysis/analysis.py
        if self.valid_values():
            shots = jt.shot_str_parser(self.value_dict["shots"].get())
            time_windows = jt.time_window_parser(self.value_dict["times"].get())
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
