# GUI/main.py #
# John Gresl 6/19/2017 #
import random
import os
from collections import OrderedDict
import Analysis.analysis
from Utilities import jtools as jt
from pyfusion import DEFAULT_CONFIG_FILE
import threading


try:
    import backend  # Python 3
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    from tkinter.messagebox import showerror
    from Analysis import analysis, point_analysis
except ImportError:
    import backend  # Python 2
    import Tkinter as tk
    from tkFileDialog import askopenfilename, asksaveasfilename
    from tkMessageBox import showerror, showinfo
    from Analysis import analysis, point_analysis
font_name = "Arial"
font = (font_name, 14)

DEFAULT_SETTINGS_DIR = os.path.join(os.path.dirname(__file__), "defaults.txt")


class ErrorWindow:
    def __init__(self, master, message):
        showerror(master=master, title="Error!", message=message)
        return

class ProcessingWindow2:
    def __init__(self, master, message, process):
        self.master = master
        self.process = process
        self.root = tk.Toplevel(master=master)
        self.sv = tk.StringVar(master=self.root, value="Processing... Please wait.")
        self.lab = tk.Label(master=self.root, textvariable=self.sv, font=(font_name,25))
        self.lab.grid(row=0, column=0, sticky=tk.N)
        cancel_button = tk.Button(master=self.root, text="Cancel", command=self.cancel)
        cancel_button.grid(row=1, column=0, sticky=tk.N)
        self.root.grab_set()
        return

    def cancel(self):
        self.process.terminate()
        self.root.destroy()
        showinfo(title="Cancelled", message="Process was terminated.")

    def launch(self):
        self.process.start()
        self.root.after(10, self.isAlive)

    def isAlive(self):
        if self.process.is_alive():
            self.root.after(100, self.isAlive)
        elif self:
            showinfo("Sucessful run", title="Finished")
            self.root.destroy()



class ProcessingWindow:
    def __init__(self, master, message):

        self.root = tk.Toplevel(master=master)
        if type(message) is str:
            self.message = tk.StringVar(master=self.root, value=message)
        else:
            self.message = message
        self.root.title("Processing.")
        self.label = tk.Label(master=self.root, textvariable=self.message, font=(font_name, 25))
        self.label.grid(row=0, column=0, sticky=tk.N)
        return

    def processing_complete(self, message="Processing complete!"):
        self.message.set(message)
        but = tk.Button(master=self.root, text="Close.", font=font, command=self.kill)
        but.grid(row=1, column=0, sticky=tk.N)
        return

    def kill(self):
        self.root.destroy()


class PromptWindow:
    def __init__(self, master, prompts, func_on_ok):
        self.root = tk.Toplevel(master=master)
        self.prompt_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.prompt_frame.grid(row=0, column=0, sticky=tk.NW)
        self.tkVars = []
        self.var_strings = []
        self.ok_pressed = False
        for (n, prompt) in enumerate(prompts):
            tk.Label(master=self.prompt_frame, text=prompt.capitalize()+":",
                     font=font).grid(row=n, column=0, sticky=tk.NE)
            var = tk.StringVar(master=self.root)
            tk.Entry(master=self.prompt_frame, font=font, textvariable=var).grid(row=n, column=1, sticky=tk.NW)
            self.tkVars.append(var)
        self.button_frame = tk.Frame(master=self.root)
        self.button_frame.grid(row=1, column=0, sticky=tk.NW)
        self.ok_button = tk.Button(master=self.button_frame, text="Ok", font=font, command=func_on_ok)
        self.ok_button.grid(row=0, column=0, sticky=tk.E)
        self.cancel_button = tk.Button(master=self.button_frame, text="Cancel", font=font, command=self.cancel)
        self.cancel_button.grid(row=0, column=1, sticky=tk.E)
        return

    def cancel(self):
        for var in self.tkVars:
            var.set("")
        self.root.destroy()
        return

    def getVars(self):
        return [i.get() for i in self.tkVars]


class PyFusionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.resizable(width=False, height=False)
        self.root.title("PyFusion GUI v. 0")
        self.root.geometry("1050x500")
        self.value_dict = OrderedDict()
        # Shot Frame (Shots, Times, Probe Arrays)
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

        # Datamining Settings (dms)
        self.dms_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.dms_frame.grid(padx=15, pady=15, row=0, column=1, sticky=tk.N + tk.E)
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

        # Buttons
        self.settings_buttons_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.settings_buttons_frame.grid(row=0, column=2, sticky=tk.N, padx=15, pady=15)

        self.save_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Save Settings",
                                              font=(font_name, 13), width=14,
                                              command=self.save_settings)
        self.save_settings_button.grid(row=0, column=0, sticky=tk.N)
        self.load_settings_button = tk.Button(master=self.settings_buttons_frame,
                                              text="Load From File",
                                              font=(font_name, 13), width=14,
                                              command=self.load_settings)
        self.load_settings_button.grid(row=1, column=0, sticky=tk.N)
        self.restore_defaults_button = tk.Button(master=self.settings_buttons_frame,
                                                 text="Restore Defaults",
                                                 font=(font_name, 13), width=14,
                                                 command=self.restore_defaults)
        self.restore_defaults_button.grid(row=2, column=0, sticky=tk.N)

        self.run_clustering_button = tk.Button(master=self.settings_buttons_frame,
                                               text="Cluster",
                                               font=(font_name, 13), width=14,
                                               command=self.run_clustering)
        self.run_clustering_button.grid(row=3, column=0, sticky=tk.N)

        self.run_point_analysis_button = tk.Button(master=self.settings_buttons_frame,
                                                   text="Point Analysis",
                                                   font=(font_name, 13), width=14,
                                                   command=self.run_point_analysis)
        self.run_point_analysis_button.grid(row=4, column=0, sticky=tk.N)

        # Save everything to a dictionary of tkinter StringVar's.
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

    def clustering_complete(self):

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

    def load_values(self, f):
        with open(f) as vals:
            lines = vals.readlines()
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

    def load_settings(self):
        fname = askopenfilename(initialdir=os.path.dirname(__file__),
                                filetypes=(("Text File (*.txt)", "*.txt"), ("All Files", "*.*")))
        if fname == "":
            return None
        try:
            self.load_values(fname)
        except:
            ErrorWindow(self.root, "Incorrect file format.")
        return None

    def save_settings(self):
        if self.valid_values():
            fname = asksaveasfilename(initialdir=os.path.dirname(__file__),
                                      filetypes=(("Text File (*.txt)", "*.txt"), ("All Files", "*.*")))
            if fname == "":
                return None
            backend.save_values(self.value_dict, fname)
            return None

    def restore_defaults(self):
        self.load_values(DEFAULT_SETTINGS_DIR)
        return None

    def run_clustering(self):

        def callback():
            # A = self.settings_to_analysis_object()
            # A.run_analysis()
            root = tk.Tk()
            sv = tk.StringVar(master=root, value="Now clustering.\nPlease wait.")
            label = tk.Label(master=root, textvariable=sv, font=font)
            label.grid(row=0, column=0)
            import time
            time.sleep(2)
            #window.processing_complete("Clustering complete!")
            sv.set("Clustering complete!")
            button = tk.Button(master=root, text="Close", command=root.destroy)
            button.grid(row=1, column=0)
            return


        #sv = tk.StringVar(master=self.root, value="Now clustering.\nPlease wait.")
        #win = ProcessingWindow(master=self.root, message=sv)
        #win.root.grab_set()

        t = threading.Thread(target=callback)
        t.start()
        return None

    def run_point_analysis(self):
        # A window needs to popup asking the user for a frequency and time.
        # Verify both frequency and time are within our analysis windows.
        # Can only be done after the analysis has been performed. (Grey out button before that??)
        # but we'll ignore the latter for now.
        def run():
            shot, time_window, freq, time = win.getVars()
            if not jt.valid_int_from_str(shot):
                ErrorWindow(master=win.root, message="Shot entry is not valid.")
            elif not jt.in_tkStringVar_array(shot, self.value_dict["shots"]):
                ErrorWindow(master=win.root, message="Shot entry is not within the analysis range.")
            elif not jt.valid_window(time_window):
                ErrorWindow(master=win.root, message="Time window entry is not valid.")
            elif not jt.window_subset(jt.time_window_parser(time_window),
                                      jt.time_window_parser(self.value_dict["times"].get())):
                ErrorWindow(master=win.root, message="Time window must be contained within the analysis time window.")
            elif not jt.valid_float_from_str(freq):
                ErrorWindow(master=win.root, message="Frequency entry is not valid.")
            elif not jt.valid_float_from_str(time):
                ErrorWindow(master=win.root, message="Time entry is not valid.")
            elif not jt.t_in_window(time, time_window):
                ErrorWindow(master=win.root, message="Time entry must be within the time window.")
            else:
                win.root.destroy()
            return
        win = PromptWindow(master=self.root,
                           prompts=["Shot", "Time Window (ms)", "Frequency (khz)", "Time (ms)"],
                           func_on_ok=run)
        win.root.grab_set()
        self.root.wait_window(win.root)
        shot, time_window, freq, time = win.getVars()
        time_window = jt.time_window_parser(time_window)
        A = self.settings_to_analysis_object()
        point_analysis.point_analysis(A, shot, time_window, time, freq,
                                      self.value_dict["probe_array"].get(), clustarr="all")
        return

    def settings_to_analysis_object(self):
        # Takes the current settings and creates an Analysis object from Analysis/analysis.py
        # A1 = Analysis(shots=shots, time_windows=time_windows, probes="DIIID_toroidal_mag", markersize=15,
        #               datamining_settings={'n_clusters': 8, 'n_iterations': 20, 'start': 'k_means', 'verbose': 0,
        #                                    'method': 'EM_VMM', "seeds": None})
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
            seeds = int(self.value_dict["seed"].get())
            datamining_settings = {'n_clusters': n_clusters, 'n_iterations': n_iterations, 'start': start, 'verbose': 0,
                                   'method': method, "seeds": seeds}
            A = analysis.Analysis(shots=shots, time_windows=time_windows, probes=probes, markersize=15,
                                  datamining_settings=datamining_settings, n_cpus=n_cpus)
            return A
        return None

if __name__ == '__main__':
    window1 = PyFusionWindow()
    window1.start()
    pass
