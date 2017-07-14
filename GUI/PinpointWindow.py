
# tkinter
try:  # Will work with python 3
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror
except ImportError:  # Will work with python 2
    import Tkinter as tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    from tkMessageBox import showerror, showinfo

from CONSTANTS import *
from Utilities import jtools as jt
import matplotlib.pyplot as plt
from ErrorWindow import *
import threading


class PinpointWindow:
    def __init__(self, master, defaults, pf_window, previous_analysis=None):
        self.previous_analysis = previous_analysis
        self.pf_window = pf_window
        self.root = tk.Toplevel(master=master)
        self.root.geometry("350x350")
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
