# GUI/main.py #
# John Gresl 6/19/2017 #
import random
import os
from collections import OrderedDict
try:
    import backend # Python 3
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror
except ImportError:
    from GUI import backend # Python 2
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    from tkMessageBox import showerror
font_name = "Arial"
font = (font_name, 14)


class ErrorWindow:
    def __init__(self, master, message):
        showerror(master=master, title="Error!", message=message)
        return


class PyFusionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("PyFusion GUI v. 0")
        self.root.geometry("1050x700")
        self.value_dict = OrderedDict()
        # Shot Frame (Shots, Times, Probe Arrays)
        self.shot_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.shot_frame.grid(padx=15, pady=15, row=0, column=0, sticky=tk.N + tk.W)
        self.shot_info = tk.Label(master=self.shot_frame,
                                  text="Shot Info",
                                  font=(font_name, 20)).grid(row=0, column=0, columnspan=2, sticky=tk.NSEW)

        self.shot_label = tk.Label(master=self.shot_frame,
                                   text="Shots:",
                                   font=font).grid(row=1, sticky=tk.E)
        self.shot_help_label = tk.Label(master=self.shot_frame,
                                        text="Enter shot numbers and/or shot ranges separated" +
                                             "\nby commas. For example, 159243,159248-159255.",
                                        font=(font_name, 9)).grid(row=2, column=0, columnspan=2)
        self.shot_var = tk.StringVar(master=self.shot_frame)
        self.shots = tk.Entry(master=self.shot_frame,
                              font=font,
                              textvariable=self.shot_var).grid(row=1, column=1, padx=5)

        self.time_label = tk.Label(master=self.shot_frame,
                                   text="Times:",
                                   font=font).grid(row=3, sticky=tk.E)
        self.time_help_label = tk.Label(master=self.shot_frame,
                                        text="Enter a time range in ms. For example, 300-1400.",
                                        font=(font_name, 9)).grid(row=4, column=0, columnspan=2)
        self.time_var = tk.StringVar(master=self.shot_frame)
        self.times = tk.Entry(master=self.shot_frame,
                              font=font,
                              textvariable=self.time_var).grid(row=3, column=1, padx=5)

        self.probes_label = tk.Label(master=self.shot_frame,
                                     text="Probe Array:",
                                     font=font).grid(row=5, sticky=tk.E)
        self.probes_help_label = tk.Label(master=self.shot_frame,
                                          text="Enter a probe array name. For example, DIIID_toroidal_mag." +
                                          "\nThese are defined in pyfusion.cfg",
                                          font=(font_name, 9)).grid(row=6, column=0, columnspan=2)
        self.probe_var = tk.StringVar(master=self.shot_frame)
        self.probes = tk.Entry(master=self.shot_frame,
                               font=font,
                               textvariable=self.probe_var).grid(row=5, column=1, padx=5)

        # Datamining Settings (dms)
        self.dms_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.dms_frame.grid(padx=15, pady=15, row=0, column=1, sticky=tk.N + tk.E)
        self.dms_info = tk.Label(master=self.dms_frame,
                                 text="Datamining Settings",
                                 font=(font_name, 20)).grid(row=0, column=0, columnspan=4, sticky=tk.NSEW)
        self.ncpus_label = tk.Label(master=self.dms_frame,
                                    text="n_cpus:",
                                    font=font).grid(row=1, column=0, sticky=tk.E)
        self.ncpus_help_label = tk.Label(master=self.dms_frame,
                                         text="Enter number of CPU's to use.",
                                         font=(font_name, 9)).grid(row=2, column=0, columnspan=2, sticky=tk.NSEW)
        self.ncpus_var = tk.StringVar(master=self.dms_frame)
        self.ncpus_entry = tk.Entry(master=self.dms_frame,
                                    width=9,
                                    font=font,
                                    textvariable=self.ncpus_var).grid(row=1, column=1, padx=5, sticky=tk.W)
        self.nclusters_label = tk.Label(master=self.dms_frame,
                                        text="n_clusters:",
                                        font=font).grid(row=3, column=0, sticky=tk.E)
        self.nclusters_help_label = tk.Label(master=self.dms_frame,
                                             text="Enter number of clusters to look for.",
                                             font=(font_name, 9)).grid(row=4, column=0, columnspan=2, sticky=tk.NSEW)
        self.nclusters_var = tk.StringVar(master=self.dms_frame)
        self.nclusters_entry = tk.Entry(master=self.dms_frame,
                                        width=9,
                                        font=font,
                                        textvariable=self.nclusters_var).grid(row=3, column=1, padx=5, sticky=tk.W)
        self.niter_label = tk.Label(master=self.dms_frame,
                                    text="n_iterations:",
                                    font=font).grid(row=5, column=0, sticky=tk.E)
        self.niter_help_label = tk.Label(master=self.dms_frame,
                                         text="Enter number of clustering iterations.",
                                         font=(font_name, 9)).grid(row=6, column=0, columnspan=2, sticky=tk.NSEW)
        self.niter_var = tk.StringVar(master=self.dms_frame)
        self.niter_entry = tk.Entry(master=self.dms_frame,
                                    font=font,
                                    width=9,
                                    textvariable=self.niter_var).grid(row=5, column=1, padx=5, sticky=tk.W)
        self.start_method_label = tk.Label(master=self.dms_frame,
                                           text="Start Method:",
                                           font=font).grid(row=7, column=0, sticky=tk.E)
        self.start_method_help_label = tk.Label(master=self.dms_frame,
                                                text="Enter starting clustering method.",
                                                font=(font_name, 9)).grid(row=8, column=0, columnspan=2, sticky=tk.NSEW)
        self.start_method_var = tk.StringVar(master=self.dms_frame)
        self.start_method_entry = tk.Entry(master=self.dms_frame,
                                           font=font,
                                           width=9,
                                           textvariable=self.start_method_var).grid(row=7,
                                                                                    column=1,
                                                                                    padx=5,
                                                                                    sticky=tk.W)
        self.method_label = tk.Label(master=self.dms_frame,
                                     text="Method:",
                                     font=font).grid(row=9, column=0, sticky=tk.E)
        self.method_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter final clustering method.",
                                          font=(font_name, 9)).grid(row=10, column=0, columnspan=2, sticky=tk.NSEW)
        self.method_var = tk.StringVar(master=self.dms_frame)
        self.method_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.method_var).grid(row=9, column=1, padx=5, sticky=tk.W)
        self.freq_range_label = tk.Label(master=self.dms_frame,
                                         text="Freq. range:",
                                         font=font).grid(row=11, column=0, sticky=tk.E)
        self.freq_range_help_label = tk.Label(master=self.dms_frame,
                                              text="Enter frequency range in kHz.",
                                              font=(font_name, 9)).grid(row=12, column=0, columnspan=2, sticky=tk.NSEW)
        self.freq_range_var = tk.StringVar(master=self.dms_frame)
        self.freq_range_entry = tk.Entry(master=self.dms_frame,
                                         font=font,
                                         width=9,
                                         textvariable=self.freq_range_var).grid(row=11, column=1, padx=5, sticky=tk.W)

        self.seed_label = tk.Label(master=self.dms_frame,
                                   text="Seed:",
                                   font=font).grid(row=1, column=3, sticky=tk.E)
        self.seed_help_label = tk.Label(master=self.dms_frame,
                                        text="Enter clustering seed.",
                                        font=(font_name, 9)).grid(row=2, column=3, columnspan=2, sticky=tk.NSEW)
        self.seed_var = tk.StringVar(master=self.dms_frame)
        self.seed_entry = tk.Entry(master=self.dms_frame,
                                   font=font,
                                   width=9,
                                   textvariable=self.seed_var).grid(row=1, column=4, padx=5, sticky=tk.W)
        self.random_seed_button = tk.Button(master=self.dms_frame,
                                            text="Random Seed",
                                            font=(font_name, 9),
                                            command=self.random_seed,
                                            bd=3).grid(row=3, column=4, sticky=tk.NSEW)

        self.npeaks_label = tk.Label(master=self.dms_frame,
                                     text="n_peaks:",
                                     font=font).grid(row=5, column=3, sticky=tk.E)
        self.npeaks_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter number of peaks to find.",
                                          font=(font_name, 9)).grid(row=6, column=3, columnspan=2, sticky=tk.NSEW)
        self.npeaks_var = tk.StringVar(master=self.dms_frame)
        self.npeaks_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.npeaks_var).grid(row=5, column=4, padx=5, sticky=tk.W)

        self.cutoff_label = tk.Label(master=self.dms_frame,
                                     text="Cutoff By:",
                                     font=font).grid(row=7, column=3, sticky=tk.E)
        self.cutoff_help_label = tk.Label(master=self.dms_frame,
                                          text="Enter cutoff by item.",
                                          font=(font_name, 9)).grid(row=8, column=3, columnspan=2, sticky=tk.NSEW)
        self.cutoff_var = tk.StringVar(master=self.dms_frame)
        self.cutoff_entry = tk.Entry(master=self.dms_frame,
                                     font=font,
                                     width=9,
                                     textvariable=self.cutoff_var).grid(row=7, column=4, padx=5, sticky=tk.W)
        self.cutoff_val_label = tk.Label(master=self.dms_frame,
                                         text="Cutoff Value:",
                                         font=font).grid(row=9, column=3, sticky=tk.E)
        self.cutoff_val_help_label = tk.Label(master=self.dms_frame,
                                              text="Enter filter cutoff value.",
                                              font=(font_name, 9)).grid(row=10, column=3, columnspan=2, sticky=tk.NSEW)
        self.cutoff_val_var = tk.StringVar(master=self.dms_frame)
        self.cutoff_val_entry = tk.Entry(master=self.dms_frame,
                                         font=font,
                                         width=9,
                                         textvariable=self.cutoff_val_var).grid(row=9, column=4, padx=5, sticky=tk.W)
        self.filter_item_label = tk.Label(master=self.dms_frame,
                                          text="Filter Items:",
                                          font=font).grid(row=11, column=3, sticky=tk.E)
        self.filter_item_help_label = tk.Label(master=self.dms_frame,
                                               text="Enter filter items.",
                                               font=(font_name, 9)).grid(row=12, column=3, columnspan=2, sticky=tk.NSEW)
        self.filter_item_var = tk.StringVar(master=self.dms_frame)
        self.filter_item_entry = tk.Entry(master=self.dms_frame,
                                          font=font,
                                          width=9,
                                          textvariable=self.filter_item_var).grid(row=11, column=4, padx=5, sticky=tk.W)

        # Buttons
        self.settings_buttons_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.settings_buttons_frame.grid(row=0, column=2, sticky=tk.N)

        self.save_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Save Settings",
                                              font=(font_name, 13), width=14,
                                              command=self.save_settings).grid(row=0, column=0, sticky=tk.N)
        self.load_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Load From File",
                                              font=(font_name, 13), width=14,
                                              command=self.load_settings).grid(row=1, column=0, sticky=tk.N)
        self.restore_defaults_button = tk.Button(master=self.settings_buttons_frame,
                                                 text="Restore Defaults",
                                                 font=(font_name, 13), width=14,
                                                 command=self.restore_defaults).grid(row=2, column=0, sticky=tk.N)

        return

    def start(self):
        self.restore_defaults()
        self.root.mainloop()
        return None

    def random_seed(self):
        self.seed_var.set(str(random.choice(range(200, 1000))))
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

    def values_to_dict(self):
        self.value_dict["shots"] = self.shot_var.get()
        self.value_dict["times"] = self.time_var.get()
        self.value_dict["probe_array"] = self.probe_var.get()
        self.value_dict["n_cpus"] = self.ncpus_var.get()
        self.value_dict["n_clusters"] = self.nclusters_var.get()
        self.value_dict["n_iterations"] = self.niter_var.get()
        self.value_dict["start"] = self.start_method_var.get()
        self.value_dict["method"] = self.method_var.get()
        self.value_dict["freq_range"] = self.freq_range_var.get()
        self.value_dict["seed"] = self.seed_var.get()
        self.value_dict["n_peaks"] = self.npeaks_var.get()
        self.value_dict["cutoff_by"] = self.cutoff_var.get()
        self.value_dict["cutoff_value"] = self.cutoff_val_var.get()
        self.value_dict["filter_items"] = self.filter_item_var.get()
        return None

    def load_settings(self):
        fname = askopenfilename(initialdir=os.getcwd(),
                                filetypes=(("Text File (*.txt)", "*.txt"), ("All Files", "*.*")))
        if fname == "": return None
        try:
            self.value_dict = backend.load_values(fname)
            self.update_values()
        except:
            ErrorWindow(self.root, "Unable to open file.")
        return None

    def save_settings(self):
        fname = asksaveasfilename(initialdir=os.getcwd(),
                                  filetypes=(("Text File (*.txt)", "*.txt"), ("All Files", "*.*")))
        if fname == "": return None
        self.values_to_dict()
        backend.save_values(self.value_dict, fname)
        return None

    def restore_defaults(self):
        self.value_dict = backend.load_values("defaults.txt")
        self.update_values()
        return None

if __name__ == '__main__':
    window1 = PyFusionWindow()
    window1.start()

    pass
