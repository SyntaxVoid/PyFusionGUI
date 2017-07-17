#!/bin/bash
#SBATCH -p short
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 12.9
#SBATCH --mem 115G
#SBATCH -o /home/%u/PyFusionGUI/PyFusionGUI/SLURM/PyFusionGUI-%j.out
#SBATCH --export=ALL
set -e
echo "Starting job on worker node"
/fusion/usc/opt/python/2.7.11/bin/python2.7 C:\Users\John Gresl\OneDrive\Documents\PyFusionGUI\PyFusionGUI\SLURM\temp.py
