# We import the Analysis module
from PyFusionGUI.Analysis.analysis import *
# Initialize all our settings
shots = [159243, 159244, 159245, 159246]
time_windows = [300, 1400]
probes = "DIIID_toroidal_mag"
datamining_settings = {"verbose": 0, "n_clusters": 16, "start": "k_means",
                       "n_iterations": 20, "seeds": 743,
                       "method": "EM_VMM"}
fft_settings = {"upper_freq": 250, "cutoff_by": "sigma_eq",
                "ave_kappa_cutoff": 70, "filter_item": "EM_VMM_kappas",
                "lower_freq": 50, "n_pts": 20}
n_cpus = 1
# Create the datamining object and run the datamining
DM1 = DataMining(shots=shots, time_windows=time_windows, probes=probes,
                 datamining_settings=datamining_settings,
                 fft_settings=fft_settings, n_cpus=n_cpus)
# Run the analysis on the datamining object above
AN1 = Analysis(DM=DM1)
# Save the analysis object
AN1.save("/cscratch/cluster_example.ANobj")
