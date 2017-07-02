# Analysis/analysis.py #
# John Gresl 6/20/2017 #

import copy
import os
import itertools
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["axes.linewidth"] = 4.0
import pyfusion as pf
import pyfusion.clustering as clust
import pyfusion.clustering.extract_features_scans as ext
from Utilities import jtools as jt
import pickle
from PyFusionGUI import *

class OutOfOrderError(Exception):
    pass

class AnalysisError(Exception):
    pass

def stft_pickle_workaround(input_data):
    # This looks a little funny. Because of how python treats multiprocessing, any function
    # using mp must be at the highest scope level (not inside a class) to operate correctly.
    return copy.deepcopy(input_data[0].get_stft(input_data[1]))


def stft_ece_pickle_workaround(input_data):
    # This looks a little funny. Because of how python treats multiprocessing, any function
    # using mp must be at the highest level (not within a class) to operate correctly.
    return copy.deepcopy(input_data[0].get_stft_ece(input_data[1]))


def plot_seperate_clusters(A, clust_arr, ax=None, doplot=True, dosave=None):
    # Inputs:
    #   A: Result from Analysis class (A.run_analysis() must have already been run.
    #   clust_arr: Array of the clusters we want to plot. eg: [1,4,6] will plot clusters 1,4 and 6
    #   ax: if ax is supplied, will plot it on the specified axes
    # Outputs:
    #   A graph.
    if clust_arr == "all":
        clust_arr = np.unique(A.z.cluster_assignments)

    class CycledList:
        def __init__(self, arr):
            self.arr = arr
            return

        def __getitem__(self, key):
            return self.arr[np.mod(key, len(self.arr))]

    plot_colors = CycledList(["silver", "darkorchid", "royalblue", "red", "chartreuse", "gold", "olivedrab",
                              "mediumspringgreen", "lightseagreen", "darkcyan", "deepskyblue",
                              "c", "sienna", "m", "mediumvioletred", "lightsalmon"])
    if ax is None:
        plt.figure(figsize=(11, 8.5), dpi=100, facecolor="w", edgecolor="k")
        plt.specgram(A.results[0][2][0, :], NFFT=1024, Fs=1./np.mean(np.diff(A.results[0][3])),
                     noverlap=128, xextent=[A.results[0][3][0], A.results[0][3][-1]])
        for cl in clust_arr:
            mask = (A.z.cluster_assignments == cl)
            plt.plot(A.z.feature_obj.misc_data_dict["time"][mask],
                     A.z.feature_obj.misc_data_dict["freq"][mask],
                     color=plot_colors[cl], marker="o", linestyle="None",
                     markersize=A.markersize)
        # Cause I'm lazy.
        if dosave is not None and "toroidal" in dosave.lower():
            plt.title("Shot 159243 Toroidal Array")
        if dosave is not None and "poloidal" in dosave.lower():
            plt.title("Shot 159243 Poloidal Array")
        plt.xlabel("Time (ms)")
        plt.ylabel("Freq (kHz)")
        plt.xlim([750, 850])
        plt.ylim([45, 250])
        if doplot:
            plt.show()
        if dosave is not None:
            plt.savefig(dosave)
    else:
        ax.specgram(A.results[0][2][0, :], NFFT=1024, Fs=1. / np.mean(np.diff(A.results[0][3])),
                    noverlap=128, xextent=[A.results[0][3][0], A.results[0][3][-1]])
        for cl in clust_arr:
            mask = (A.z.cluster_assignments == cl)
            ax.plot(A.z.feature_obj.misc_data_dict["time"][mask],
                    A.z.feature_obj.misc_data_dict["freq"][mask],
                    color=plot_colors[cl], marker="o", linestyle="None",
                    markersize=A.markersize)
    return

class Analysis2:
    # Updated Analysis object. These methods were separated from the DataMining object
    # so that there could be a clear intermediary step between the datamining step and
    # the analysis step. The keywords _from_pickle and _pickle_data should be used very
    # carefuly. As far as I know, python is unable to have separate constructors for the
    # same class, which is why I use the if-else blocks and the _from_pickle keyword.
    def __init__(self, DM=None, _from_pickle=False, _pickle_data=None):
        if not _from_pickle:
            if DM is None:
                raise ValueError("DataMining object DM cannot be none!")
            self.DM = DM
            self.results, self.feature_object, self.z = self.run_analysis()
        else:
            self.results = _pickle_data["self"]["results"]
            self.feature_object = _pickle_data["self"]["feature_object"]
            self.z = _pickle_data["self"]["z"]
            self.DM = DataMining(_from_pickle=True, _pickle_data=_pickle_data["DM"])
        return

    def __repr__(self):
        return "<<Analysis object for {} >>".format(self.DM.__repr__())

    def run_analysis(self):
        # Returns analysis
        func = stft_pickle_workaround
        tmp_data_iter = itertools.izip(itertools.repeat(self.DM),
                                       self.DM.shot_info["shots"],
                                       self.DM.shot_info["time_windows"])
        if self.DM.n_cpus > 1:
            # We can process each shot separately using different processing cores.
            pool = Pool(processes=self.DM.n_cpus, maxtasksperchild=3)
            results = pool.map(func, tmp_data_iter)
            pool.close()
            pool.join()
        else:
            results = map(func, tmp_data_iter)
        start = True
        instance_array = 0
        misc_data_dict = 0
        for n, res in enumerate(results):
            if res[0] is not None:
                if start:
                    instance_array = copy.deepcopy(res[0])
                    misc_data_dict = copy.deepcopy(res[1])
                    start = False
                else:
                    instance_array = np.append(instance_array, res[0], axis=0)
                    for i in misc_data_dict.keys():
                        misc_data_dict[i] = np.append(misc_data_dict[i], res[1][i], axis=0)
            else:
                raise AnalysisError("Shot {} has failed!".format(self.DM.shot_info["shots"][n]))
        feature_object = clust.feature_object(instance_array=instance_array, misc_data_dict=misc_data_dict,
                                              instance_array_amps=+misc_data_dict["mirnov_data"])
        z = feature_object.cluster(**self.DM.datamining_settings)
        return results, feature_object, z

    @classmethod
    def restore(cls, filename):
        if not os.path.exists(filename):
            raise OSError("File ({}) does not exist!".format(filename))
        with open(filename, "rb") as pick:
            data = pickle.load(pick)
        return cls(_from_pickle=True, _pickle_data=data)

    def save(self, filename=None):
        # Saves the current instance variables as a pickled object to "filename".
        # Since there are several different data structures within an analysis object,
        # they will be compressed into a single dictionary object and then pickled.
        if filename is None:
            probes = self.DM.shot_info["probes"]
            if probes == "DIIID_toroidal_mag":
                pr = "TOR"
            elif probes == "DIIID_poloidal322_mag":
                pr = "POL"
            elif probes == "ECEF_array":
                pr = "ECE"
            elif probes == "ECEF_array_red":
                pr = "ECE_REDUCED"
            else:
                pr = probes
            local_filename = str(self.DM.shot_info["shots"][0]) + "_" + \
                             jt.time_window_to_filelike_str(self.DM.shot_info["time_windows"][0]) + "_" + \
                             pr + ".ANobj"
            filename = os.path.join(PICKLE_SAVE_DIR, local_filename)
            print(filename)
        with open(filename, "wb") as pick:
            pickle.dump({"self": {"results": self.results,
                                  "feature_object": self.feature_object,
                                  "z": self.z}, "DM": self.DM.__dict__}, pick)
        return

    def return_plots(self):
        fontsize   = 35  # FixMe: More robust definition of fontsize... Not sure what to do now.
        markersize = 15  # FixMe: More robust definition of markersize
        plot_colors = jt.CycledList(["#ff0000", "#ff9400", "#ffe100", "#bfff00",
                                     "#2aff00", "#00ffa9", "#00f6ff", "#0090ff",
                                     "#0033ff", "#8700ff", "#cb00ff", "#ff00f2",
                                     "#ff006a", "#631247", "#a312f7", "#a3f2f7"])
        n_shots = len(self.DM.shot_info["shots"])
        nrows, ncols = jt.squareish_grid(n_shots, swapxy=True)
        figure1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        figure2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
        axesf1 = axes1.flatten() if n_shots > 1 else np.array([axes1], dtype=object)
        axesf2 = axes2.flatten() if n_shots > 1 else np.array([axes2], dtype=object)
        big_axes1 = figure1.add_subplot(111, frameon=False)
        big_axes2 = figure2.add_subplot(111, frameon=False)
        big_axes1.spines["top"].set_color("none")
        big_axes1.spines["left"].set_color("none")
        big_axes1.spines["right"].set_color("none")
        big_axes1.spines["bottom"].set_color("none")
        #big_axes1.tick_params(top="off", bottom="off", left="off", right="off")
        big_axes2.spines["top"].set_color("none")
        big_axes2.spines["left"].set_color("none")
        big_axes2.spines["right"].set_color("none")
        big_axes2.spines["bottom"].set_color("none")
        #big_axes2.tick_params(top="off", bottom="off", left="off", right="off")
        big_axes1.set_xticklabels([" "])
        big_axes2.set_xticklabels([" "])
        big_axes1.set_yticklabels([" "])
        big_axes2.set_yticklabels([" "])
        big_axes1.set_xlabel("Time (ms)")
        big_axes1.set_ylabel("Freq (kHz)")
        big_axes2.set_xlabel("Time (ms)")
        big_axes2.set_ylabel("Freq (kHz)")
        for current_axes1, current_axes2, shot, result in \
                zip(axesf1, axesf2, self.DM.shot_info["shots"], self.results):
            assignments = self.z.cluster_assignments
            details = self.z.cluster_details["EM_VMM_kappas"]
            shot_details = self.z.feature_obj.misc_data_dict["shot"]
            print("SHOT DETAILS:", shot_details)
            time_base = result[3]
            signal = result[2][0, :]
            dt = np.mean(np.diff(time_base))
            current_axes1.specgram(signal, NFFT=1024, Fs=1./dt, noverlap=128, xextent=[time_base[0], time_base[-1]])
            current_axes2.specgram(signal, NFFT=1024, Fs=1./dt, noverlap=128, xextent=[time_base[0], time_base[-1]])
            for assignment in np.unique(assignments):
                mask = (assignments==assignment)*(shot_details==shot)
                if np.sum(mask) > 1 and np.mean(details[assignment, :]) > 5:  # FixMe: What does this do?
                    current_axes1.plot(self.z.feature_obj.misc_data_dict["time"][mask],
                                       self.z.feature_obj.misc_data_dict["freq"][mask],
                                       "o", markersize=markersize,
                                       color=plot_colors[assignment])
        for i in range(n_shots):
            shot = str(self.DM.shot_info["shots"][i])
            axesf1[i].set_xlim(self.DM.shot_info["time_windows"][i])
            axesf2[i].set_xlim(self.DM.shot_info["time_windows"][i])
            axesf1[i].set_ylim([0, 250])
            axesf2[i].set_ylim([0, 250])
            tx, ty = jt.text_location(self.DM.shot_info["time_windows"][i], [0, 250])
            axesf1[i].text(tx, ty, shot, bbox={"facecolor": "green", "alpha": 0.90}, fontsize=fontsize)
            axesf2[i].text(tx, ty, shot, bbox={"facecolor": "green", "alpha": 0.90}, fontsize=fontsize)
        figure1.subplots_adjust(hspace=0, wspace=0)
        figure2.subplots_adjust(hspace=0, wspace=0)
        # figure1.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize-10)
        # figure2.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize - 10)
        # figure1.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize-10)
        # figure2.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize - 10)
        figure1.tight_layout()
        figure2.tight_layout()
        return ((figure1, axes1), (figure2, axes2))


class DataMining:
    # Updated version of original Analysis class, renamed to Datamining since all it does is
    # fetch data about the probes. All of the Analysis features have been ported over the the
    # NEW Analysis class in an attempt to separate the code and make it more transparent.
    # New features include ability to save and load DataMining objects from previous run cycles.
    # This saves CPU time by not having to perform datamining every time. Also faster, more
    # pythonic, and a lower memory usage. The keywords '_from_pickle' and '_pickle_data' should
    # be used very carefully. As far as I know, python is unable to have separate constructors
    # for the same class, which is why I use the if-else blocks and the _from_pickle keyword.
    def __init__(self, shots=None, time_windows=None, device="DIIID", probes="DIIID_toroidal_mag",
                 fft_settings=None, datamining_settings=None, n_cpus=1,
                 _from_pickle=False, _pickle_data=None):
        if not _from_pickle:
            # This executes the 'normal' construction of an Analysis object.
            if shots is None:
                shots = 159243
            if type(shots) == int:
                shots = [shots]
            if time_windows is None:
                time_windows = list(itertools.repeat([300, 1400], len(shots)))
            elif type(time_windows) is not list:
                time_windows = list(itertools.repeat(time_windows, len(shots)))
            elif type(time_windows[0]) is not list:
                time_windows = [time_windows]
            self.shot_info = {"shots": shots, "time_windows": time_windows, "device": device, "probes": probes}
            self.fft_settings = fft_settings if fft_settings is not None else \
                {"n_pts": 8, "lower_freq": 10, "upper_freq": 250, "cutoff_by": "sigma_eq",
                 "ave_kappa_cutoff": 25, "filter_item": "EM_VMM_kappas"}
            self.datamining_settings = datamining_settings if datamining_settings is not None else \
                {'n_clusters': 16, 'n_iterations': 20, 'start': 'k_means', 'verbose': 0, 'method': 'EM_VMM', "seeds": None}
            self.n_cpus = n_cpus
            self.mags = self.return_mags()
            self.raw_ffts = self.return_raw_ffts()
            self.raw_mirnov_datas = self.return_raw_mirnov_datas()
            self.raw_times = self.return_raw_times()
        else:
            # This executes the construction of an Analysis object from a previously saved
            # pickle file. That file must have been saved using the save method.
            try:
                self.shot_info = _pickle_data["shot_info"]
                self.fft_settings = _pickle_data["fft_settings"]
                self.datamining_settings = _pickle_data["datamining_settings"]
                self.n_cpus = _pickle_data["n_cpus"]
                self.mags = _pickle_data["mags"]
                self.raw_ffts = _pickle_data["raw_ffts"]
                self.raw_mirnov_datas = _pickle_data["raw_mirnov_datas"]
                self.raw_times = _pickle_data["raw_times"]
            except:
                raise pickle.PickleError("Incorrect pickle file format.")
        return

    def __repr__(self):
        # Used to display something user friendly when executing print(<analysis_object>)
        return "<<DataMining object for shots {}>>".format(sorted(self.shot_info["shots"]))

    @classmethod
    def restore(cls, filename):
        # Creates and returns an analysis object from a pickled file created
        # using the class' save method.
        if not os.path.exists(filename):
            raise OSError("File ({}) does not exist!".format(filename))
        with open(filename, "rb") as pick:
            data = pickle.load(pick)
        return cls(_from_pickle=True, _pickle_data=data)

    def save(self, filename=None):
        # Saves the current instance variables as a pickled object to "filename".
        # Since there are several different data structures within an analysis object,
        # they will be compressed into a single dictionary object and then pickled. If
        # filename is None, will create a custom filename based off of the FIRST shot,
        # FIRST time window and probe name
        # FixMe: Remove the comment below if program works fine.
        # data = {"shot_info": self.shot_info, "fft_settings": self.fft_settings,
        #         "datamining_settings": self.datamining_settings, "n_cpus": self.n_cpus,
        #         "mags": self.mags, "raw_ffts": self.raw_ffts,
        #         "raw_mirnov_datas": self.raw_mirnov_datas, "raw_times": self.raw_times}
        if filename is None:
            probes = self.shot_info["probes"]
            if probes == "DIIID_toroidal_mag":
                pr = "TOR"
            elif probes == "DIIID_poloidal322_mag":
                pr = "POL"
            elif probes == "ECEF_array":
                pr = "ECE"
            elif probes == "ECEF_array_red":
                pr = "ECE_REDUCED"
            else:
                pr = probes
            local_filename = str(self.shot_info["shots"][0]) + "_" + \
                             jt.time_window_to_filelike_str(self.shot_info["time_windows"][0]) + "_" + \
                             pr + ".DMobj"
            filename = os.path.join(PICKLE_SAVE_DIR, local_filename)
            print(filename)
        with open(filename, "wb") as pick:
            pickle.dump(self.__dict__, pick)
        return

    def return_mags(self):
        # Returns the magnitudes of every shot and time window in the form of a dictionary.
        # Output is formatted like: {"159243": magnitudes, "159244": magnitudes, ... }
        out = {}
        dev = pf.getDevice(self.shot_info["device"])
        for sh,tw in zip(self.shot_info["shots"], self.shot_info["time_windows"]):
            print(sh, tw)
            out[str(sh)] = dev.acq.getdata(sh, self.shot_info["probes"]).reduce_time(tw)
        return out

    def return_raw_ffts(self):
        # Returns the FFT's of every shot and in the form of a dictionary
        # Output is formatted like: {"159243": FFT, "159244": FFT, ... }
        # Requires that magnitudes have already been stored within self.mags!!!!
        if len(self.mags.keys()) == 0:
            raise OutOfOrderError("Magnitudes must be calculated before FFTs!\n"+
                                  "Run self.mags = self.return_mags().")
        out = {}
        for sh in self.shot_info["shots"]:
            mag = self.mags[str(sh)]
            out[str(sh)] = mag.generate_frequency_series(1024, 256)
        return out

    def return_raw_mirnov_datas(self):
        # Returns the raw mirnov data of every shot in the form of a dictionary
        # Output is formatted like: {"159243": mirnov, "159244": mirnov, ... }
        # Requires that FFTs have already been stored within self.raw_ffts!!!!
        if len(self.raw_ffts.keys()) == 0:
            raise OutOfOrderError("Raw FFTs must be calculated before raw mirnov data!\n"+
                                  "Run self.raw_ffts = self.return_raw_ffts().")
        out = {}
        for sh in self.shot_info["shots"]:
            out[str(sh)] = self.raw_ffts[str(sh)].signal
        return out

    def return_raw_times(self):
        # Returns the raw times of every shot in the form of a dictionary
        # Output is formatted like: {"159243": mirnov, "159244": mirnov, ... }
        # Requires that FFTs have already been stored within self.raw_ffts!!!!
        if len(self.raw_ffts.keys()) == 0:
            raise OutOfOrderError("Raw FFTs must be calculated before raw mirnov data!\n"+
                                  "Run self.raw_ffts = self.return_raw_ffts().")
        out = {}
        for sh in self.shot_info["shots"]:
            out[str(sh)] = self.raw_ffts[str(sh)].timebase
        return out

    def get_stft(self, shot):
        # Calculates the short time fourier transform (STFT) of a given shot in the current
        # analysis object. This function must be dependent on shot since we use map and
        # multiprocessing at a later point in the script since each shot can be processed
        # independently.
        magi = self.mags[str(shot)]
        data_ffti = self.raw_ffts[str(shot)]
        good_indices = ext.find_peaks(data_ffti, **self.fft_settings)
        rel_data = ext.return_values(data_ffti.signal, good_indices)
        n = len(ext.return_non_freq_dependent(data_ffti.frequency_base, good_indices))
        misc_data_dict = {"time": ext.return_time_values(data_ffti.timebase, good_indices),
                          "freq": ext.return_non_freq_dependent(data_ffti.frequency_base, good_indices),
                          "shot": np.ones(n, dtype=int) * shot,
                          "mirnov_data": +rel_data}
        rel_data_angles = np.angle(rel_data)
        diff_angles = (np.diff(rel_data_angles)) % (2. * np.pi)
        diff_angles[diff_angles > np.pi] -= (2. * np.pi)
        z = ext.perform_data_datamining(diff_angles, misc_data_dict, self.datamining_settings)
        instance_array_cur, misc_data_dict_cur = \
            ext.filter_by_kappa_cutoff(z, ax=None, **self.fft_settings)
        instance_array = np.array(instance_array_cur)
        misc_data_dict = misc_data_dict_cur
        return instance_array, misc_data_dict, magi.signal, magi.timebase

    def get_stft_wrapper(self, input_data):
        # We have to use a wrapper like this in order for multiprocessing to function correctly
        return copy.deepcopy(self.get_stft(input_data[0]))



class Analysis:

    def return_cluster_plot(self):
        # Returns a matplotlib plot object for use in tkinter
        fontsize = 35
        plot_colors = {1: "#ff0000", 2: "#ff9400", 3: "#ffe100", 4: "#bfff00", 5: "#2aff00",
                       6: "#00ffa9", 7: "#00f6ff", 8: "#0090ff", 9: "#0033ff", 10: "#8700ff",
                       11: "#cb00ff", 12: "#ff00f2", 13: "#ff006a"}
        mpl.rcParams["axes.linewidth"] = 4.0
        markersize = self.markersize
        nshots = len(self.shots)
        if nshots > 2:
            nrows, ncols = jt.squareish_grid(nshots, swapxy=True)
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
            self.axf = self.ax.flatten()
            self.fig2, self.ax2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
            self.axf2 = self.ax2.flatten()
            for cur_ax, cur_ax2, shot, tmp in zip(self.axf, self.axf2, self.shots, self.results):
                assign = self.z.cluster_assignments
                details = self.z.cluster_details["EM_VMM_kappas"]
                shot_details = self.z.feature_obj.misc_data_dict["shot"]
                time_base = tmp[3]
                sig = tmp[2]
                dt = np.mean(np.diff(time_base))
                tmp_sig = sig[0, :]
                self.im = cur_ax.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                          noverlap=128, xextent=[time_base[0], time_base[-1]])
                self.im = cur_ax2.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                           noverlap=128, xextent=[time_base[0], time_base[-1]])
                for i in np.unique(assign):
                    mask = (assign == i) * (shot_details == shot)
                    if np.sum(mask) > 1 and np.mean(details[i, :]) > 5:
                        if i not in plot_colors:
                            self.pl = cur_ax.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                                  self.z.feature_obj.misc_data_dict['freq'][mask],
                                                  'o', markersize=markersize)
                            plot_colors[i] = self.pl[0].get_color()
                        else:
                            self.pl = cur_ax.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                                  self.z.feature_obj.misc_data_dict['freq'][mask],
                                                  'o', markersize=markersize, color=plot_colors[i])
            print(plot_colors)
            tmp = len(self.time_windows)
            for _ in range(tmp):
                shot = str(self.shots[_])
                self.axf[_].set_xlim(self.time_windows[_])
                self.axf2[_].set_xlim(self.time_windows[_])
                self.axf[_].set_ylim([0, 250])
                self.axf2[_].set_ylim([0, 250])
                self.axf[_].text(680, 200, shot, bbox=dict(facecolor="green", alpha=0.90), fontsize=fontsize)
                self.axf2[_].text(680, 200, shot, bbox=dict(facecolor="green", alpha=0.90), fontsize=fontsize)
        elif nshots == 1:
            # No subplots. Just a single plot.
            self.fig = plt.figure(1)
            self.fig2 = plt.figure(2)
            assign = self.z.cluster_assignments
            details = self.z.cluster_details["EM_VMM_kappas"]
            shot_details = self.z.feature_obj.misc_data_dict["shot"]
            shot = self.shots[0]
            res = self.results[0]
            time_base = res[3]
            sig = res[2]
            dt = np.mean(np.diff(time_base))
            tmp_sig = sig[0, :]
            plt.figure(1)
            self.im = plt.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                   noverlap=128, xextent=[time_base[0], time_base[-1]])
            plt.figure(2)
            self.im = plt.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                   noverlap=128, xextent=[time_base[0], time_base[-1]])
            for i in np.unique(assign):
                plt.figure(2)
                mask = (assign == i) * (shot_details == shot)
                if np.sum(mask) > 1 and np.mean(details[i, :]) > 5:
                    if i not in plot_colors:
                        self.pl = plt.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                           self.z.feature_obj.misc_data_dict['freq'][mask],
                                           'o', markersize=markersize)
                        plot_colors[i] = self.pl[0].get_color()
                    else:
                        self.pl = plt.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                           self.z.feature_obj.misc_data_dict['freq'][mask],
                                           'o', markersize=markersize, color=plot_colors[i])
            plt.figure(1)
            plt.xlim(self.time_windows[0])
            plt.ylim([50, 150])
            plt.yticks(np.arange(50, 150, 5.0))
            x0 = self.time_windows[0][0] + 0.35 * (self.time_windows[0][1] - self.time_windows[0][0])
            y0 = 50 + 0.9 * (150 - 50)
            plt.text(x0, y0, str(shot) + ": " + self.probes, bbox=dict(facecolor="green", alpha=0.9), fontsize=fontsize)
            plt.figure(2)
            plt.xlim(self.time_windows[0])
            plt.ylim([50, 150])
            plt.yticks(np.arange(50, 150, 5.0))
            plt.text(x0, y0, str(shot) + ": " + self.probes, bbox=dict(facecolor="green", alpha=0.9), fontsize=fontsize)
        self.fig.subplots_adjust(hspace=0, wspace=0)
        self.fig2.subplots_adjust(hspace=0, wspace=0)
        self.fig.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize - 10)
        self.fig.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize - 10)
        self.fig2.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize - 10)
        self.fig2.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize - 10)
        return self.fig, self.fig2



    def plot_clusters(self):
        fontsize = 35
        plot_colors = {1: "#ff0000", 2: "#ff9400", 3: "#ffe100", 4: "#bfff00", 5: "#2aff00",
                       6: "#00ffa9", 7: "#00f6ff", 8: "#0090ff", 9: "#0033ff", 10: "#8700ff",
                       11: "#cb00ff", 12: "#ff00f2", 13: "#ff006a"}
        mpl.rcParams["axes.linewidth"] = 4.0
        markersize = self.markersize
        nshots = len(self.shots)
        if nshots > 2:
            nrows, ncols = jt.squareish_grid(nshots, swapxy=True)
            self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
            self.axf = self.ax.flatten()
            self.fig2, self.ax2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
            self.axf2 = self.ax2.flatten()
            for cur_ax, cur_ax2, shot, tmp in zip(self.axf, self.axf2, self.shots, self.results):
                assign = self.z.cluster_assignments
                details = self.z.cluster_details["EM_VMM_kappas"]
                shot_details = self.z.feature_obj.misc_data_dict["shot"]
                time_base = tmp[3]
                sig = tmp[2]
                dt = np.mean(np.diff(time_base))
                tmp_sig = sig[0, :]
                self.im = cur_ax.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                          noverlap=128, xextent=[time_base[0], time_base[-1]])
                self.im = cur_ax2.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                           noverlap=128, xextent=[time_base[0], time_base[-1]])
                for i in np.unique(assign):
                    mask = (assign == i) * (shot_details == shot)
                    if np.sum(mask) > 1 and np.mean(details[i, :]) > 5:
                        if i not in plot_colors:
                            self.pl = cur_ax.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                                  self.z.feature_obj.misc_data_dict['freq'][mask],
                                                  'o', markersize=markersize)
                            plot_colors[i] = self.pl[0].get_color()
                        else:
                            self.pl = cur_ax.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                                  self.z.feature_obj.misc_data_dict['freq'][mask],
                                                  'o', markersize=markersize, color=plot_colors[i])
            print(plot_colors)
            tmp = len(self.time_windows)
            for _ in range(tmp):
                shot = str(self.shots[_])
                self.axf[_].set_xlim(self.time_windows[_])
                self.axf2[_].set_xlim(self.time_windows[_])
                self.axf[_].set_ylim([0, 250])
                self.axf2[_].set_ylim([0, 250])
                self.axf[_].text(680, 200, shot, bbox=dict(facecolor="green", alpha=0.90), fontsize=fontsize)
                self.axf2[_].text(680, 200, shot, bbox=dict(facecolor="green", alpha=0.90), fontsize=fontsize)
        elif nshots == 1:
            # No subplots. Just a single plot.
            self.fig = plt.figure(1)
            self.fig2 = plt.figure(2)
            assign = self.z.cluster_assignments
            details = self.z.cluster_details["EM_VMM_kappas"]
            shot_details = self.z.feature_obj.misc_data_dict["shot"]
            shot = self.shots[0]
            res = self.results[0]
            time_base = res[3]
            sig = res[2]
            dt = np.mean(np.diff(time_base))
            tmp_sig = sig[0, :]
            plt.figure(1)
            self.im = plt.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                   noverlap=128, xextent=[time_base[0], time_base[-1]])
            plt.figure(2)
            self.im = plt.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                                   noverlap=128, xextent=[time_base[0], time_base[-1]])
            for i in np.unique(assign):
                plt.figure(2)
                mask = (assign == i) * (shot_details == shot)
                if np.sum(mask) > 1 and np.mean(details[i, :]) > 5:
                    if i not in plot_colors:
                        self.pl = plt.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                           self.z.feature_obj.misc_data_dict['freq'][mask],
                                           'o', markersize=markersize)
                        plot_colors[i] = self.pl[0].get_color()
                    else:
                        self.pl = plt.plot(self.z.feature_obj.misc_data_dict['time'][mask],
                                           self.z.feature_obj.misc_data_dict['freq'][mask],
                                           'o', markersize=markersize, color=plot_colors[i])
            plt.figure(1)
            plt.xlim(self.time_windows[0])
            plt.ylim([50, 150])
            plt.yticks(np.arange(50, 150, 5.0))
            x0 = self.time_windows[0][0] + 0.35 * (self.time_windows[0][1] - self.time_windows[0][0])
            y0 = 50 + 0.9 * (150 - 50)
            plt.text(x0, y0, str(shot) + ": " + self.probes, bbox=dict(facecolor="green", alpha=0.9), fontsize=fontsize)
            plt.figure(2)
            plt.xlim(self.time_windows[0])
            plt.ylim([50, 150])
            plt.yticks(np.arange(50, 150, 5.0))
            plt.text(x0, y0, str(shot) + ": " + self.probes, bbox=dict(facecolor="green", alpha=0.9), fontsize=fontsize)
        self.fig.subplots_adjust(hspace=0, wspace=0)
        self.fig2.subplots_adjust(hspace=0, wspace=0)
        self.fig.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize - 10)
        self.fig.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize - 10)
        self.fig2.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=fontsize - 10)
        self.fig2.text(0.1, 0.5, "Freq (kHz)", va="center", rotation="vertical", fontsize=fontsize - 10)
        self.fig.canvas.draw()
        self.fig.show()
        self.fig2.canvas.draw()
        self.fig2.show()
        return

    def plot_diagnostics(self, shot, time_window, t0, f0, idx="", doplot=True, dosave=None, clust_arr=None):
        # Plots amplitude vs. phase graphs with a spectrogram aswell. t0 and f0 are the time and frequency
        # you would like the phases and amplitudes for. If dosave is equal to a filename string, the plot
        # will be saved to that file.
        pi = np.pi
        fft = self.raw_ffts[str(shot)]
        raw_mirnov = fft.signal
        raw_times = fft.timebase
        raw_freqs = fft.frequency_base

        nt, t_actual = jt.find_closest(raw_times, t0)
        nf, f_actual = jt.find_closest(raw_freqs, f0)
        print("Requested t={}ms. Got t={}ms. dt={}ms".format(t0, t_actual, abs(t0 - t_actual)))
        print("Requested f={}kHz. Got f={}kHz. df={}kHz".format(f0, f_actual, abs(f0 - f_actual)))

        complex_amps = []
        tmp = raw_mirnov[nt]
        for prb in tmp:
            complex_amps.append(prb[nf])
        amps = jt.complex_mag_list(complex_amps)
        phases = np.angle(complex_amps)
        positions = []
        if idx.lower() == "tor":
            positions = [20., 67., 97., 127., 132., 137., 157., 200., 247., 277., 307., 312., 322., 340.]
        elif idx.lower() == "pol":
            positions = [000.0, 018.4, 036.0, 048.7, 059.2, 069.6, 078.0, 085.1, 093.4, 100.7, 107.7,
                         114.9, 121.0, 129.2, 143.6, 165.3, 180.1, 195.0, 216.3, 230.8, 238.9, 244.9,
                         253.5, 262.1, 271.1, 279.5, 290.6, 300.6, 311.8, 324.2, 341.9]
        elif idx.lower() == "ece3":
            positions = [36 * 0, 36 * 1, 36 * 2, 36 * 3, 36 * 4, 36 * 5, 36 * 6, 36 * 7, 36 * 8, 36 * 9]
        tmp = self.results[0]
        time_base = tmp[3]
        sig = tmp[2]
        dt = np.mean(np.diff(time_base))
        tmp_sig = sig[0, :]

        plt.figure(num=None, figsize=(11, 8.5), dpi=100, facecolor="w", edgecolor="k")
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'STIXGeneral'
        mpl.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)

        ax1.plot(positions, amps, "k*-", linewidth=2)
        ax1.set_xlabel("Probe Positions ($^\circ$)", fontsize=16)
        ax1.set_ylabel("Amplitudes", fontsize=16)
        ax1.set_xticks(np.arange(0, 360 + 1, 60))
        ax1.set_xlim([0, 360])
        ax1.grid()

        ax2.plot(positions, phases, "k*-", linewidth=2)
        ax2.set_xlabel("Probe Positions ($^\circ$)", fontsize=16)
        ax2.set_ylabel("Phase", fontsize=16)
        ax2.set_xlim([0, 360])
        ax2.set_xticks(np.arange(0, 360 + 1, 60))
        ax2.set_yticks([-pi, -3 * pi / 4, -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi])
        ax2.set_yticklabels(["$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "$0$",
                             r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$\pi$"])
        ax2.set_ylim([-pi, pi])
        ax2.grid()

        ax3.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                     noverlap=128, xextent=[time_base[0], time_base[-1]])
        if idx.lower() == "tor":
            plot_seperate_clusters(self, clust_arr, ax3)
        if idx.lower() == "ece3":
            plot_seperate_clusters(self, clust_arr, ax3)
        elif idx.lower() == "pol":
            pass
        ax3.set_xlabel("Time (ms)", fontsize=16)
        ax3.set_ylabel("Freq (kHz)", fontsize=16)

        ax3.plot([t0, t0], [45, 250], "k")
        ax3.plot(time_window, [f0, f0], "k")
        ax3.set_xlim(time_window)
        ax3.set_ylim([45, 250])

        plt.suptitle("Shot 159243 ({}) at t = {} ms, f = {} kHz".format(idx, t_actual, f_actual), fontsize=24)
        plt.subplots_adjust(wspace=0.4)
        if doplot:
            plt.show()
        if dosave is not None:
            plt.savefig(dosave)
        return

    # ECE #
    def get_stft_ece(self, shot):
        magi = self.mags[str(shot)]
        data_ffti = self.raw_ffts[str(shot)]
        good_indices = ext.find_peaks(data_ffti, **self.fft_settings)
        rel_data = ext.return_values(data_ffti.signal, good_indices)
        tmp = len(ext.return_non_freq_dependent(data_ffti.frequency_base, good_indices))
        misc_data_dict = {"time": ext.return_time_values(data_ffti.timebase, good_indices),
                          "freq": ext.return_non_freq_dependent(data_ffti.frequency_base, good_indices),
                          "shot": np.ones(tmp, dtype=int) * shot,
                          "mirnov_data": +rel_data}
        print("######################" * 50)
        rel_data_angles = np.angle(rel_data)
        diff_angles = (np.diff(rel_data_angles)) % (2. * np.pi)
        diff_angles[diff_angles > np.pi] -= (2. * np.pi)
        diff_amps = np.abs(np.diff(jt.complex_mag_list(rel_data)))
        # This is where ECE is different. Use magnitudes as clustering instead of angles
        z = ext.perform_data_datamining(diff_amps, misc_data_dict, self.datamining_settings)
        instance_array_cur, misc_data_dict_cur = \
            ext.filter_by_kappa_cutoff(z, **self.fft_settings)
        instance_array = np.array(instance_array_cur)
        misc_data_dict = misc_data_dict_cur
        return instance_array, misc_data_dict, magi.signal, magi.timebase

    def get_stft_ece_wrapper(self, input_data):
        return copy.deepcopy(self.get_stft_ece(input_data[0]))

    def run_analysis_ece(self, method="stft", savefile=None):
        if method == "stft":
            func = stft_ece_pickle_workaround
        else:
            func = None
        # tmp_data_iter = itertools.izip(itertools.repeat(self),itertools.izip(self.shots,self.time_windows))
        tmp_data_iter = itertools.izip(itertools.repeat(self), self.shots, self.time_windows)
        if self.n_cpus > 1:
            pool = Pool(processes=self.n_cpus, maxtasksperchild=3)
            self.results = pool.map(func, tmp_data_iter)
            pool.close()
            pool.join()
        else:
            self.results = map(func, tmp_data_iter)
        start = True
        instance_array = 0
        misc_data_dict = 0
        for n, res in enumerate(self.results):
            if res[0] is not None:
                if start:
                    instance_array = copy.deepcopy(res[0])
                    misc_data_dict = copy.deepcopy(res[1])
                    start = False
                else:
                    instance_array = np.append(instance_array, res[0], axis=0)
                    for i in misc_data_dict.keys():
                        misc_data_dict[i] = np.append(misc_data_dict[i], res[1][i], axis=0)
            else:
                print("One shot may have failed!")
        self.feature_object = clust.feature_object(instance_array=instance_array, misc_data_dict=misc_data_dict,
                                                   instance_array_amps=+misc_data_dict["mirnov_data"])
        self.z = self.feature_object.cluster(**self.datamining_settings)
        if savefile is not None:
            self.feature_object.dump_data(savefile)
        return


if __name__ == '__main__':
    # # Example of how to use these classes
    shots = 159243
    time_windows = [300, 700]
    probes = "DIIID_toroidal_mag"
    # ## DataMining
    # # Create the datamining object. Creating it will automatically perform the datamining,
    # # however it will take a little bit of time (on the order of minutes).
    # DM1 = DataMining(shots=shots, time_windows=time_windows, probes=probes)
    # # Saving to a default directory, no keyword filename required.
    # # DM1.save()
    # # Saving to a custom directory.
    # DM1.save(filename="TESTDMSAVE.DMobj")
    # # Restoring
    # DM2 = DataMining.restore(filename="TESTDMSAVE.DMobj")
    #
    # ## Analysis
    # # Create the analysis object from the previously defined DataMining object. Creating it will
    # # automatically perform the analysis, however it will take a little bit of time (on the order
    # # of minutes).
    # AN1 = Analysis2(DM=DM1)
    # # Saving to a default directory, no keyword filename required.
    # # AN1.save()
    # # Saving to a custom directory.
    # AN1.save(filename="TESTANSAVE.ANobj")
    # # Restoring
    # AN2 = Analysis2.restore(filename="TESTANSAVE.ANobj")
    DM1 = DataMining(shots=shots, time_windows=time_windows, probes=probes)
    AN1 = Analysis2(DM=DM1)
    plot1, plot2 = AN1.return_plots()
    plt.show()
