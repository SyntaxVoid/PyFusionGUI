import os.path
from getpass import getuser


# Here is where all of the important PyFusion file locations will be stored.
# They are automatically imported when importing PyFusionGUI. You must make
# sure the PARENT directory of PyFusion is listed in your Python path to import.
USERNAME = getuser()
IRIS_CSCRATCH_DIR = r"/cscratch/{u}/".format(u=USERNAME)

PROJECT_ROOT_DIR = os.path.dirname(__file__)
ANALYSIS_DIR = os.path.join(PROJECT_ROOT_DIR, "Analysis")
GUI_DIR = os.path.join(PROJECT_ROOT_DIR, "GUI")
PYFUSION_DIR = os.path.join(PROJECT_ROOT_DIR, "pyfusion")
UTILITIES_DIR = os.path.join(PROJECT_ROOT_DIR, "Utilities")
SLURM_DIR = os.path.join(PROJECT_ROOT_DIR, "SLURM")