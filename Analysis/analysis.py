# Native Python
import os
import copy
import itertools
from multiprocessing import Pool
import pickle

# Anaconda
import numpy as np

# PyFusion
import pyfusion as pf
import pyfusion.clustering as clust
import pyfusion.clustering.extract_features_scans as ext
from PyFusionGUI import *

# My Stuff
from Utilities import jtools as jt
OutOfOrderError = jt.OutOfOrderException
AnalysisError = jt.AnalysisError

pi = np.pi
PLOT_COLORS = jt.CycledList(["#ff0000", "#ff9400", "#ffe100", "#bfff00",
                             "#2aff00", "#00ffa9", "#00f6ff", "#0090ff",
                             "#0033ff", "#8700ff", "#cb00ff", "#ff00f2",
                             "#ff006a", "#631247", "#a312f7", "#a3f2f7"])

def stft_pickle_workaround(input_data):
    # This looks a little funny. Because of how python treats multiprocessing, any function
    # using mp must be at the highest scope level (not inside a class) to operate correctly.
    #return copy.deepcopy(input_data[0].get_stft(input_data[1]))
    return input_data[0].get_stft(input_data[1])
def mag_pickle_workaround(input_data):
    #return copy.deepcopy(input_data[0].get_mag(input_data[1], input_data[2], input_data[3]))
    return input_data[0].get_mag(input_data[1], input_data[2], input_data[3])

def fft_pickle_workaround(input_data):
    #return copy.deepcopy(input_data[0].get_raw_fft(input_data[1]))
    return input_data[0].get_raw_fft(input_data[1])

def mirnov_pickle_workaround(input_data):
    #return copy.deepcopy(input_data[0].get_raw_mirnov(input_data[1]))
    return input_data[0].get_raw_mirnov(input_data[1])
def times_pickle_workaround(input_data):
    #return copy.deepcopy(input_data[0].get_raw_time(input_data[1]))
    return input_data[0].get_raw_time(input_data[1])

def stft_ece_pickle_workaround(input_data):
    # This looks a little funny. Because of how python treats multiprocessing, any function
    # using mp must be at the highest level (not within a class) to operate correctly.
    #return copy.deepcopy(input_data[0].get_stft_ece(input_data[1]))
    return input_data[0].get_stft_ece(input_data[1])


def _big_axes(ax, xlabel="Time (ms)", ylabel="Freq (kHz)"):
    # Hack that modifies an axes that we will lay overtop our other subplots,
    # allowing us to easily place common x and y labels, which otherwise isn't clear.
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="x", pad=15)
    ax.tick_params(axis="y", pad=23)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return None


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
            elif type(time_windows) is list:
                time_windows = list(itertools.repeat(time_windows, len(shots)))
            elif type(time_windows[0]) is not list:
                time_windows = [time_windows]
            self.shot_info = {"shots": shots, "time_windows": time_windows, "device": device, "probes": probes}
            self.fft_settings = fft_settings if fft_settings is not None else \
                {"n_pts": 8, "lower_freq": 10, "upper_freq": 250, "cutoff_by": "sigma_eq",
                 "ave_kappa_cutoff": 25, "filter_item": "EM_VMM_kappas"}
            self.datamining_settings = datamining_settings if datamining_settings is not None else \
                {'n_clusters': 16, 'n_iterations': 20, 'start': 'k_means',
                 'verbose': 0, 'method': 'EM_VMM', "seeds": None}
            self.n_cpus = n_cpus

            mag_iter = itertools.izip(itertools.repeat(self),
                                      self.shot_info["shots"],
                                      self.shot_info["time_windows"],
                                      itertools.repeat(self.shot_info["probes"]))
            self.mags = self.mp_acquire(func=mag_pickle_workaround, iter=mag_iter)
            fft_iter = itertools.izip(itertools.repeat(self),
                                      self.shot_info["shots"])

            self.raw_ffts = self.mp_acquire(func=fft_pickle_workaround, iter=fft_iter)
            raw_mirnov_iter = itertools.izip(itertools.repeat(self),
                                             self.shot_info["shots"])  # Add shots to avoid endless loops
            self.raw_mirnov_datas = self.mp_acquire(func=mirnov_pickle_workaround, iter=raw_mirnov_iter)
            raw_times_iter = itertools.izip(itertools.repeat(self),
                                            self.shot_info["shots"])
            self.raw_times = self.mp_acquire(times_pickle_workaround, iter=raw_times_iter)
            # self.mags = self.return_mags()
            # self.raw_ffts = self.return_raw_ffts()
            # self.raw_mirnov_datas = self.return_raw_mirnov_datas()
            # self.raw_times = self.return_raw_times()
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
        from time import strftime
        time_string = strftime("%d-%m-%Y--%H-%M-%S")
        if filename is None:
            local_filename = "PyFusionGUI_from_{}".format(time_string)
            filename = os.path.join(IRIS_CSCRATCH_DIR, local_filename)
        with open(filename, "wb") as pick:
            pickle.dump(self.__dict__, pick)
        return

    @staticmethod
    def get_mag(shot, time_window, probe):
        dev = pf.getDevice("DIIID") # We have to create a new device object for every shot otherwise MDSPlus
                                    # will throw a fit about faulty connections.
        return dev.acq.getdata(shot, probe).reduce_time(time_window)

    def get_raw_fft(self, shot):
        return self.mags[str(shot)].generate_frequency_series(1024, 256)

    def get_raw_mirnov(self, shot):
        return self.raw_ffts[str(shot)].signal

    def get_raw_time(self, shot):
        return self.raw_ffts[str(shot)].timebase

    def mp_acquire(self, func, iter):
        out = {}
        if self.n_cpus > 1:
            pool = Pool(processes=self.n_cpus, maxtasksperchild=3)
            ans = pool.map(func, iter)
            pool.close()
            pool.join()
        else:
            ans = map(func, iter)
        for shot, result in zip(self.shot_info["shots"], ans):
            out[str(shot)] = result
        return out

    def get_stft(self, shot, clustering_parameter="phase"):
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
        if self.datamining_settings["amplitude"]:
            divisor = np.sum(np.abs(misc_data_dict['mirnov_data']),axis=1)
            misc_data_dict['mirnov_data'] = np.abs(misc_data_dict['mirnov_data'] / divisor[:, np.newaxis])
        else:
            pass
        rel_data_angles = np.angle(rel_data)
        diff_angles = (np.diff(rel_data_angles)) % (2. * np.pi)
        diff_angles[diff_angles > np.pi] -= (2. * np.pi)
        z = ext.perform_data_datamining(diff_angles, misc_data_dict, self.datamining_settings)
        instance_array_cur, misc_data_dict_cur = \
            ext.filter_by_kappa_cutoff(z, **self.fft_settings)
        instance_array = np.array(instance_array_cur)
        misc_data_dict = misc_data_dict_cur
        return instance_array, misc_data_dict, magi.signal, magi.timebase

    def get_stft_wrapper(self, input_data):
        # We have to use a wrapper like this in order for multiprocessing to function correctly
        return copy.deepcopy(self.get_stft(input_data[0]))


class Analysis:
    # Updated Analysis object. These methods were separated from the DataMining object
    # so that there could be a clear intermediary step between the datamining step and
    # the analysis step. The keywords _from_pickle and _pickle_data should be used very
    # carefully. As far as I know, python is unable to have separate constructors for the
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
        # Returns results, feature object, and z of the analysis object. Called in __init__.
        func = stft_pickle_workaround
        tmp_data_iter = itertools.izip(itertools.repeat(self.DM),
                                       self.DM.shot_info["shots"])
        if self.DM.n_cpus > 1:
            # We can process each shot separately using different processing cores.
            pool = Pool(processes=self.DM.n_cpus, maxtasksperchild=3)
            results = pool.map(func, tmp_data_iter)
            pool.close()
            pool.join()
        else:
            results = map(func, tmp_data_iter)
        start = True
        instance_array = {}
        misc_data_dict = {}
        for n, res in enumerate(results):
            if res[0] != None:
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
        if misc_data_dict is None:
            return None, None, None
        feature_object = clust.feature_object(instance_array=instance_array, misc_data_dict=misc_data_dict,
                                              instance_array_amps=+misc_data_dict["mirnov_data"])
        z = feature_object.cluster(amplitude=True, **self.DM.datamining_settings)
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
        from time import strftime
        time_string = strftime("%d-%m-%Y--%H-%M-%S")
        if filename is None:
            local_filename = "PyFusionGUI_from_{}".format(time_string)
            filename = os.path.join(IRIS_CSCRATCH_DIR, local_filename)
        with open(filename, "wb") as pick:
            pickle.dump({"self": {"results": self.results,
                                  "feature_object": self.feature_object,
                                  "z": self.z}, "DM": self.DM.__dict__}, pick)
        return

    def show_plots(self):
        import matplotlib.pyplot as plt
        self.return_specgrams()
        plt.show()
        return

    def return_specgrams(self):
        import matplotlib.pyplot as plt  # These imports are here since this file will be imported into an
        import matplotlib as mpl         # environment where matplotib cannot be imported
        mpl.rcParams["axes.linewidth"] = 4.0
        n_shots = len(self.DM.shot_info["shots"])
        nrows, ncols = jt.squareish_grid(n_shots, swapxy=True)
        fontsize = np.ceil(35/(ncols**0.25))  # FixMe: More robust definition of fontsize... Not sure what to do now.
        markersize = 5  # FixMe: More robust definition of markersize
        figure1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, edgecolor="k", facecolor="w")
        figure2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, edgecolor="k", facecolor="w")
        axesf1 = axes1.flatten() if n_shots > 1 else np.array([axes1], dtype=object)
        axesf2 = axes2.flatten() if n_shots > 1 else np.array([axes2], dtype=object)
        _big_axes(figure1.add_subplot(111, frameon=False))
        _big_axes(figure2.add_subplot(111, frameon=False))
        for current_axes1, current_axes2, shot, result in \
                zip(axesf1, axesf2, self.DM.shot_info["shots"], self.results):
            assignments = self.z.cluster_assignments
            details = self.z.cluster_details["EM_VMM_kappas"]
            shot_details = self.z.feature_obj.misc_data_dict["shot"]
            time_base = result[3]
            signal = result[2][0, :]
            dt = np.mean(np.diff(time_base))
            current_axes1.specgram(signal, NFFT=1024, Fs=1./dt, noverlap=128, xextent=[time_base[0], time_base[-1]])
            current_axes2.specgram(signal, NFFT=1024, Fs=1./dt, noverlap=128, xextent=[time_base[0], time_base[-1]])
            for assignment in np.unique(assignments):
                mask = (assignments == assignment)*(shot_details == shot)
                if np.sum(mask) > 1 and np.mean(details[assignment, :]) > 5:  # FixMe: What does this do?
                    current_axes1.plot(self.z.feature_obj.misc_data_dict["time"][mask],
                                       self.z.feature_obj.misc_data_dict["freq"][mask],
                                       "o", markersize=markersize,
                                       color=PLOT_COLORS[assignment])
        shots = self.DM.shot_info["shots"]
        time_windows = self.DM.shot_info["time_windows"]
        freq_window = [self.DM.fft_settings["lower_freq"], self.DM.fft_settings["upper_freq"]]
        for i in range(n_shots):
            shot = str(shots[i])
            axesf1[i].set_xlim(time_windows[i])
            axesf2[i].set_xlim(time_windows[i])
            axesf1[i].set_ylim(freq_window)
            axesf2[i].set_ylim(freq_window)
            tx, ty = jt.text_location(time_windows[i], freq_window)
            axesf1[i].text(tx, ty, shot, bbox={"facecolor": "green", "alpha": 0.90}, fontsize=fontsize)
            axesf2[i].text(tx, ty, shot, bbox={"facecolor": "green", "alpha": 0.90}, fontsize=fontsize)
        figure1.subplots_adjust(hspace=0, wspace=0)
        figure2.subplots_adjust(hspace=0, wspace=0)
        figure1.tight_layout()
        figure2.tight_layout()
        return (figure1, axes1), (figure2, axes2)

    def return_pinpoint_plots(self, shot, t0, f0, time_window=None, frequency_window=None, clusters=None):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams["axes.linewidth"] = 4.0
        shot_index = self.DM.shot_info["shots"].index(int(shot))
        if clusters is None:
            clusters = []
        if time_window is None:
            time_window = [float(self.DM.shot_info["time_windows"][shot_index][0]),
                           float(self.DM.shot_info["time_windows"][shot_index][-1])]
        if frequency_window is None:
            frequency_window = [float(self.DM.fft_settings["lower_freq"]),
                                float(self.DM.fft_settings["upper_freq"])]
        fft = self.DM.raw_ffts[str(shot)]
        raw_mirnov = fft.signal
        raw_times = fft.timebase
        raw_freqs = fft.frequency_base

        nt, t_actual = jt.find_closest(raw_times, t0)
        nf, f_actual = jt.find_closest(raw_freqs, f0)

        complex_amps = []
        for probe in raw_mirnov[nt]:
            complex_amps.append(probe[nf])
        amps = jt.complex_mag_list(complex_amps)
        phases = np.angle(complex_amps)
        positions = jt.probe_positions(self.DM.shot_info["probes"])
        positions_in_degrees = True
        if positions is None:
            positions = list(range(len(phases)))
            positions_in_degrees = False

        time_base = self.results[shot_index][3]
        signal = self.results[shot_index][2]
        temp_signal = signal[0, :]

        fig = plt.figure(num=None, figsize=(11, 8.5), dpi=100, facecolor="w", edgecolor="k")
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)

        ax1.plot(positions, amps, "k*-", linewidth=2)
        self._x_axes_in_degrees(ax1, positions, positions_in_degrees)
        ax1.set_ylabel("Amplitudes", fontsize=16)
        ax1.grid()

        ax2.plot(positions, phases, "k*-", linewidth=2)
        self._x_axes_in_degrees(ax2, positions, positions_in_degrees)
        ax2.set_ylabel("Phase (rad)", fontsize=16)
        ax2.set_yticks([-pi, -3 * pi / 4, -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi])
        ax2.set_yticklabels(["$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "$0$",
                             r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$\pi$"])
        ax2.set_ylim([-pi, pi])
        ax2.grid()

        self._plot_clusters(temp_signal, time_base, clusters, ax3)
        ax3.plot([t0, t0], [frequency_window[0], frequency_window[-1]], "k")
        ax3.plot(time_window, [f0, f0], "k")
        ax3.set_xlim(time_window)
        ax3.set_ylim(frequency_window)

        plt.suptitle("Shot {} -- ({})\nt = {} ms, f = {} kHz".format(shot,
                                                                     self.DM.shot_info["probes"],
                                                                     t_actual, f_actual), fontsize=20)
        plt.subplots_adjust(wspace=0.4)
        return fig, ax1, ax2, ax3

    def _plot_clusters(self, signal, time_base, clusters, ax):
        dt = np.mean(np.diff(time_base))
        if clusters == "all":
            clusters = np.unique(self.z.cluster_assignments)
        ax.specgram(signal, NFFT=1024, Fs=1./dt, noverlap=128,
                    xextent=[time_base[0], time_base[-1]])
        for cluster in clusters:
            mask = (self.z.cluster_assignments == cluster)
            ax.plot(self.z.feature_obj.misc_data_dict["time"][mask],
                    self.z.feature_obj.misc_data_dict["freq"][mask],
                    color=PLOT_COLORS[cluster], marker="o", linestyle="None",
                    markersize=4)
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Freq (kHz)", fontsize=16)

    @staticmethod
    def _x_axes_in_degrees(ax, positions, degrees=True):
        if degrees:
            ax.set_xlabel("Probe Positions ($^\circ$)", fontsize=16)
            ax.set_xticks(np.arange(0, 360 + 1, 60))
            ax.set_xlim([0, 360])
        else:
            ax.set_xlabel("Probe Number", fontsize=16)
            ax.set_xticks(positions)
            ax.set_xlim(positions)
        return

if __name__ == '__main__':
    # # Example of how to use these classes
    shots = [159243, 159244]
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
    # AN1 = Analysis(DM=DM1)
    # # Saving to a default directory, no keyword filename required.
    # # AN1.save()
    # # Saving to a custom directory.
    # AN1.save(filename="TESTANSAVE.ANobj")
    # # Restoring
    # AN2 = Analysis.restore(filename="TESTANSAVE.ANobj")
