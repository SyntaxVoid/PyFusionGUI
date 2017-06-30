import os

# Here is where all of the important PyFusion file locations will be stored. They are automatically
# imported when importing PyFusionGUI

PYFUSIONGUI_DIR = os.path.dirname(__file__)
ANALYSIS_DIR = os.path.join(PYFUSIONGUI_DIR, "Analysis")
GUI_DIR = os.path.join(PYFUSIONGUI_DIR, "GUI")
UTILITIES_DIR = os.path.join(PYFUSIONGUI_DIR, "Utilities")


