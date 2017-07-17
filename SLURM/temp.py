from PyFusionGUI.Analysis.analysis import *
A1 = Analysis(DM=DataMining(shots=[159243], time_windows=[300, 1400], probes="DIIID_toroidal_mag",datamining_settings={'verbose': 0, 'n_clusters': 16, 'start': 'k_means', 'n_iterations': 20, 'seeds': 743, 'method': 'EM_VMM'}, fft_settings={'upper_freq': 250, 'cutoff_by': 'sigma_eq', 'ave_kappa_cutoff': 70, 'filter_item': 'EM_VMM_kappas', 'lower_freq': 50, 'n_pts': 20},n_cpus=1))
A1.save("/cscratch/John Gresl/2017-07-16-23-36-11.ANobj")
