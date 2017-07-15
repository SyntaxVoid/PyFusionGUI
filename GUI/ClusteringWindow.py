# tkinter
try:  # Will work with python 3
    import tkinter as tk
except ImportError:  # Will work with python 2
    import Tkinter as tk

from CONSTANTS import *
import subprocess
from Utilities import jtools as jt


class ClusteringWindow:
    def __init__(self, master, slurm_start_time, jobid):
        # This Toplevel window pops up when the user begins a clustering job. There are several ways to create a
        # ClusteringWindow object, but since python doesn't allow for multiple class constructors, we are stuck
        # using if/else statements instead.
        self.master = master
        self.root = tk.Toplevel(master=self.master)
        # self.root.geometry("250x200")
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
        self.root.bind("<<clustering_failed>>", self.clustering_failed)
        self.root.bind("<<slurm_clustering_complete>>", self.slurm_clustering_complete)
        self.default_wait_time = 5  # Wait 5 seconds before refreshing squeue status
        self.slurm_start_time = slurm_start_time
        self.jobid = jobid
        self.root.title("Clustering in Progress")
        self.ANobj_file = IRIS_CSCRATCH_DIR+self.slurm_start_time+".ANobj"
        self.error_file = os.path.join(SLURM_DIR, "errors.txt")
        self._cur = self.default_wait_time
        self.cancel_button = tk.Button(master=self.message_frame, text="Cancel", command=self.verify_cancel)
        self.cancel_button.grid(row=1, column=0, sticky=tk.N)
        self.total_time = 0
        return

    def start(self):
        self.root.after(1000, self.countdown)
        return

    def resize(self):
        # This method will be called whenever a widget value is changed. Will resize the window to fit widgets.
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        button_height = self.cancel_button.winfo_height()
        print("DEBUG::::: ", "{}x{}".format(label_width, label_height+button_height))
        print("DEBUG:::::", self.message.get())
        #self.root.geometry("{}x{}".format(label_width, label_height + button_height))
        return

    def set_label(self, val):
        # This method will be called to change the value of self.message
        self.message.set(val)
        self.resize()
        return

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
        self.root.destroy()
        return

    def countdown(self):
        self._cur -= 1
        self.total_time += 1
        if self._cur <= 0:
            sjobexitmod_output = subprocess.check_output("sjobexitmod -l {}".format(self.jobid), shell=True)
            exit_state = jt.get_slurm_exit_state(sjobexitmod_output)
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
        self.set_label("Waiting for worker\nnode to complete\njob # {}.\n"
                       "Checking again in\n{} seconds.\n"
                       "Total time elapsed:\n{} seconds".format(self.jobid, self._cur, self.total_time))
        self.root.after(1000, self.countdown)
        return

    def slurm_clustering_complete(self, e):
        self.root.title("SLURM Clustering Complete!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.destroy)
        # self.root.geometry("330x320")
        self.set_label("SLURM clustering complete!\nYour Analysis object\nwas saved to:\n{}\n"
                       "Total time elapsed: {} seconds"
                       .format(jt.break_path(self.ANobj_file, 23), self.total_time))
        self.cancel_button.destroy()
        ok_button = tk.Button(master=self.root, text="OK", command=self.root.destroy, font=(font_name, 18))
        ok_button.grid(row=1, column=0)
        self.master.event_generate("<<slurm_clustering_complete>>", when="tail")
        return

    def clustering_failed(self, e):
        self.root.title("Clustering Failed!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.set_label("Clustering Failed! Check\n{}\nfor more details."
                       .format(jt.break_path(os.path.join(IRIS_CSCRATCH_DIR,
                                                          "PyFusionGUI-{}.out".format(self.jobid)), 24)))
        self.label.config(fg="red")
        self.cancel_button.destroy()
        ok_button = tk.Button(master=self.buttons_frame,
                              text="OK", font=(font_name, 18),
                              command=self.root.destroy)
        ok_button.grid(row=0, column=0, sticky=tk.N)
        self.master.event_generate("<<clustering_failed>>", when="tail")
        return
