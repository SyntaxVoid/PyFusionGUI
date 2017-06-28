# Analysis/point_analysis.py #
# John Gresl 6/24/2017 #

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from Utilities import jtools as jt

pi = np.pi


def probe_positions(probe_array):
    if probe_array.lower() == "DIIID_toroidal_mag":
        return "azimuthal", [20., 67., 97., 127., 132., 137., 157., 200., 247., 277., 307., 312., 322., 340.]
    if probe_array.lower() == "DIIID_poloidal322_mag":
        return "azimuthal", [000.0, 018.4, 036.0, 048.7, 059.2, 069.6, 078.0, 085.1, 093.4, 100.7, 107.7,
                             114.9, 121.0, 129.2, 143.6, 165.3, 180.1, 195.0, 216.3, 230.8, 238.9, 244.9,
                             253.5, 262.1, 271.1, 279.5, 290.6, 300.6, 311.8, 324.2, 341.9]
    return None, None


def plot_clusters(A, clust_arr, ax=None, doplot=True, dosave=None):
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


def point_analysis(A, shot, time_window, t0, f0, probe_array, doplot=True, dosave=None, clustarr=None):
    fft = A.raw_ffts["shot"]
    raw_mirnov = fft.signal
    raw_times = fft.timebase
    raw_freqs = fft.frequency_base

    nt, t_actual = jt.find_closest(raw_times, t0)
    nf, f_actual = jt.find_closest(raw_freqs, f0)
    print("Requested t={}ms. Got t={}ms. dt={}ms".format(t0, t_actual, abs(t0-t_actual)))
    print("Requested f={}kHz. Got f={}kHz. df={}kHz".format(f0, f_actual, abs(f0-f_actual)))

    complex_amps = []
    tmp = raw_mirnov[nt]
    for prb in tmp:
        complex_amps.append(prb[nf])
    amps = jt.complex_mag_list(complex_amps)
    phases = np.angle(complex_amps)
    n_probes = len(phases)
    met, positions = probe_positions(probe_array)
    if positions is None:
        positions = list(range(n_probes))
    tmp = A.results[0]
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
    if met == "azimuthal":
        ax2.set_xlim([0, 360])
        ax2.set_xticks(np.arange(0, 360 + 1, 60))
        ax2.set_yticks([-pi, -3 * pi / 4, -pi / 2, -pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4, pi])
        ax2.set_yticklabels(["$-\pi$", r"$-\frac{3\pi}{4}$", r"$-\frac{\pi}{2}$", r"$-\frac{\pi}{4}$", "$0$",
                             r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$\pi$"])
        ax2.set_ylim([-pi, pi])

    ax2.grid()

    ax3.specgram(tmp_sig, NFFT=1024, Fs=1. / dt,
                 noverlap=128, xextent=[time_base[0], time_base[-1]])
    plot_clusters(A, clustarr, ax3)
    ax3.set_xlabel("Time (ms)", fontsize=16)
    ax3.set_ylabel("Freq (kHz)", fontsize=16)

    ax3.plot([t0, t0], [45, 250], "k")
    ax3.plot(time_window, [f0, f0], "k")
    ax3.set_xlim(time_window)
    ax3.set_ylim([45, 250])

    plt.suptitle("Shot 159243 ({})\nt = {} ms, f = {} kHz".format(probe_array, t_actual, f_actual), fontsize=24)
    plt.subplots_adjust(wspace=0.4)
    if doplot:
        plt.show()
    if dosave is not None:
        plt.savefig(dosave)
    return
