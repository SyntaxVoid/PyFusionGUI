from PyFusionGUI.Analysis.analysis import *
import matplotlib.pyplot as plt
A1 = Analysis.restore("/cscratch/cluster_example.ANobj")
plots = A1.return_specgrams()
plt.show()
