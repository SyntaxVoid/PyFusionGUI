
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
    def __init__(self, master, defaults):
        self.master = master
        self.root = tk.Toplevel(master=self.master)
        self.root.wm_protocol("WM_DELETE_WINDOW", self.verify_cancel)
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
        self.continue_button = tk.Button(master=self.button_frame, text="Continue", font=font, command=self.cont)
        self.continue_button.grid(row=0, column=0, sticky=tk.E)
        self.cancel_button = tk.Button(master=self.button_frame, text="Cancel", font=font, command=self.verify_cancel)
        self.cancel_button.grid(row=0, column=1, sticky=tk.W)
        self.root.grab_set()

        if defaults is not None:
            self.shot_var.set(defaults[0])
            self.time_window_var.set(defaults[1])
            self.freq_range_var.set(defaults[2])

        return

    def cont(self):
        self.master.event_generate("<<print>>", when="tail")
        print("boop")
        return

























    #
    #
    #
    # def analysis_complete(self, e):
    #     shot, time_window, freq_range, time, freq = self.get_vars()
    #     time_window = jt.time_window_parser(time_window)
    #     freq_range = jt.time_window_parser(freq_range)
    #     fig, \
    #     ax1,  \
    #     ax2,   \
    #     ax3 = self.AN.return_pinpoint_plots(shot=shot, t0=time, f0=freq, time_window=time_window,
    #                                        frequency_window=freq_range, clusters="all")
    #         # point_analysis.point_analysis(A=self.A, shot=shot, time_window=time_window,
    #         #                                 t0=time, f0=freq,
    #         #                                 probe_array=self.pf_window.value_dict["probe_array"].get())
    #     self.root.title("Analysis complete!")
    #     self.analysis_message.set("Analysis complete!")
    #
    #     ok = tk.Button(master=self.popup, text="OK", command = self.root.destroy, font=(font_name, 24))
    #     ok.grid(row=1, column=0, sticky=tk.N)
    #     plt.show()
    #     return
    #
    #
    # def analysis_failed(self, e):
    #     self.root.title("Analysis Failed!")
    #
    #     self.ok_button = tk.Button(master=self.button_frame, text="OK",
    #                                font=(font_name, 18), command = self.root.destroy)
    #     self.ok_button.grid(row=0, column=0, sticky=tk.N)
    #     return
    #
    # def valid_values(self):
    #     valid = True
    #     try:
    #         int(self.shot_var.get())
    #     except ValueError:
    #         valid = False
    #         ErrorWindow(self.root, "Shot entry is invalid.")
    #     if not jt.valid_window(self.time_window_var.get()):
    #         valid = False
    #         ErrorWindow(self.root, "Time window entry is invalid.")
    #     if not jt.valid_window(self.freq_range_var.get()):
    #         valid = False
    #         ErrorWindow(self.root, "Freq. range entry is invalid.")
    #     try:
    #         float(self.time_var.get())
    #     except ValueError:
    #         valid = False
    #         ErrorWindow(self.root, "Time entry is invalid.")
    #     try:
    #         float(self.freq_var.get())
    #     except ValueError:
    #         valid = False
    #         ErrorWindow(self.root, "Freq. entry is invalid.")
    #     return valid
    #
    # def get_vars(self):
    #     return self.shot_var.get(), self.time_window_var.get(), \
    #            self.freq_range_var.get(), float(self.time_var.get()), float(self.freq_var.get())
    #
    #
    def verify_cancel(self):
        win = tk.Toplevel(master=self.root)
        win.resizable(width=False, height=False)
        win.grab_set()
        label = tk.Label(master=win, text="Do you really wish\nto close?", font=(font_name, 18))
        label.grid(row=0, column=0, columnspan=2, sticky=tk.N)
        yes = tk.Button(master=win, text="Yes", font=(font_name, 18), command=self.root.destroy)
        yes.grid(row=1, column=0, sticky=tk.N)
        no = tk.Button(master=win, text="No", font=(font_name, 18), command=win.destroy)
        no.grid(row=1, column=1, sticky=tk.N)
        return

def p(arg):

    return

if __name__ == "__main__":
    root = tk.Tk()
    root.bind("<<print>>", p)
    b = tk.Button(master=root, text="print", command=p)
    b.pack()
    win = PinpointWindow(master=root,defaults=None)
    root.mainloop()
    pass