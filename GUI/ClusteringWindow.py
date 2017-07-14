
# tkinter
try:  # Will work with python 3
    import tkinter as tk
    from tkinter.filedialog import asksaveasfilename
except ImportError:  # Will work with python 2
    import Tkinter as tk
    from tkFileDialog import asksaveasfilename

from CONSTANTS import *
import subprocess
from Utilities import jtools as jt
import matplotlib.pyplot as plt

class ClusteringWindow:
    def __init__(self, master, slurm_start_time=None, jobid=None, ANobj_restore=None):
        # This Toplevel window pops up when the user begins a clustering job. There are several ways to create a
        # ClusteringWindow object, but since python doesn't allow for multiple class constructors, we are stuck
        # using if/else statements instead.
        self.master = master
        self.root = tk.Toplevel(master=self.master)
        self.root.geometry("220x200")
        self.root.resizable(height=False, width=False)
        self.message_frame = tk.Frame(master=self.root)
        self.message_frame.grid(row=0, column=0, sticky=tk.N)
        self.buttons_frame = tk.Frame(master=self.root, bd=5, relief=tk.SUNKEN)
        self.buttons_frame.grid(row=1, column=0, sticky=tk.N)
        self.message = tk.StringVar(master=self.message_frame)
        self.label = tk.Label(master=self.message_frame, textvariable=self.message, font=(font_name, 24))
        self.label.grid(row=0, column=0, sticky=tk.N)
        self.root.grab_set()
        self.root.wm_protocol("WM_DELETE_WINDOW", self.verify_cancel)
        self.root.bind("<<clustering_complete>>", self.clustering_complete)
        self.root.bind("<<clustering_failed>>", self.clustering_failed)
        self.root.bind("<<slurm_clustering_complete>>", self.slurm_clustering_complete)
        if ANobj_restore is None:
            self.default_wait_time = 5  # Wait 5 seconds before refreshing squeue status
            self.AN = None
            self.slurm_start_time = slurm_start_time
            self.jobid = jobid
            self.root.title("Clustering in Progress")
            if self.slurm_start_time is None:
                self.message.set("Now clustering.\nPlease wait.")
            if slurm_start_time is not None:
                self.slurm_done_file = IRIS_CSCRATCH_DIR+self.slurm_start_time+".slurmdone"
                self.ANobj_file = IRIS_CSCRATCH_DIR+self.slurm_start_time+".ANobj"
                self.error_file = os.path.join(SLURM_DIR, "errors.txt")
                self._cur = self.default_wait_time
                self.cancel_button = tk.Button(master=self.root, text="Cancel", command=self.verify_cancel)
                self.cancel_button.grid(row=1, column=0, sticky=tk.N)
                self.root.after(1000, self.countdown)
        else:
            self.AN = ANobj_restore
            self.root.title("Analysis Object Restored")
            self.message.set("Analysis object has been restored.\nSelect an option to continue.")
            self.root.event_generate("<<clustering_complete>>", when="tail")

    def verify_cancel(self):
        win = tk.Toplevel(master=self.root)
        win.resizable(width=False, height=False)
        win.grab_set()
        label = tk.Label(master=win, text="Do you really wish\nto close?", font=(font_name, 18))
        label.grid(row=0, column=0, columnspan=2, sticky=tk.N)
        yes = tk.Button(master=win, text="Yes", font=(font_name, 18), command=self.yes_cancel)
        yes.grid(row=1, column=0, sticky=tk.N)
        no = tk.Button(master=win, text="No", font=(font_name, 18), command=win.destroy)
        no.grid(row=1, column=1, sticky=tk.N)
        return


    def yes_cancel(self):
        subprocess.check_output("scancel {}".format(self.jobid), shell=True)
        self.master.event_generate("<<clustering_failed>>", when="tail")
        self.root.destroy()
        return

    def countdown(self):
        self._cur -= 1
        if self._cur <= 0:
            sjobexitmod_output = subprocess.check_output("sjobexitmod -l {}".format(self.jobid), shell=True)
            exit_state = jt.get_slurm_exit_state(sjobexitmod_output)
            self.message.set(exit_state)
            if exit_state == "PENDING" or exit_state == "assigned":
                self._cur = self.default_wait_time
            elif exit_state == "RUNNING":
                self._cur = self.default_wait_time
            elif exit_state == "COMPLETED":
                self.root.event_generate("<<slurm_clustering_complete>>", when="tail")
                return
            elif exit_state == "FAILED":
                self.root.event_generate("<<clustering_failed>>", when="tail")
                return
            elif exit_state == "CANCELLED+":
                self.root.event_generate("<<clustering_failed>>", when="tail")
            else:
                print("UNKNOWN EXIT STATE: ({})".format(exit_state))
        self.message.set("Waiting for worker\nnode to complete\njob # {}.\nChecking again in\n{} seconds".format(self.jobid, self._cur))
        self.root.after(1000, self.countdown)
        return

    def slurm_active(self):
        squeue_output = subprocess.check_output("squeue -j {}".format(self.jobid), shell=True)
        return jt.check_slurm_for_job(squeue_output)

    def slurm_clustering_complete(self, e):
        self.root.title("SLURM Clustering Complete!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.geometry("330x350")
        self.message.set("SLURM clustering complete!\nYou can now load your\nAnalysis object file from\n{}"\
                         .format(jt.break_path(self.ANobj_file, 23)))
        self.cancel_button.destroy()
        ok_button = tk.Button(master=self.root, text="OK", command=self.root.destroy, font=(font_name, 18))
        ok_button.grid(row=1, column=0)
        self.master.event_generate("<<slurm_clustering_complete>>", when="tail")
        return

    def clustering_complete(self, e):
        # When clustering is complete, a window should pop up asking the user what they want to do.
        # Whether they want to save the actual objects, save the plots, show the plots or to close.
        size = {"height": 2, "width": 16}
        self.root.title("Analysis object loaded!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.destroy)
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
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.message.set("Clustering Failed!")
        self.label.config(fg="red")
        self.ok_button = tk.Button(master=self.buttons_frame,
                                   text="OK", font=(font_name, 18),
                                   command=self.root.destroy)
        self.ok_button.grid(row=0, column=0, sticky=tk.N)
        self.master.event_generate("<<clustering_failed>>", when="tail")
        return

    def save_objects(self):
        fname = asksaveasfilename(initialdir=IRIS_CSCRATCH_DIR,
                              filetypes=(("Analysis Object File", "*.ANobj"), ("All Files", "*.*")))
        if fname == "":
            return None
        self.AN.save(fname)
        return

    def plot_clusters(self):
        self.AN.return_specgrams()
        plt.show()
        return

