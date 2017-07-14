#!/bin/bash
#SBATCH -p short
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 5
#SBATCH --mem-per-cpu=4G
#SBATCH -o /home/%u/PyFusionGUI/PyFusionGUI/SLURM/PyFusionGUI-%j.out
#SBATCH --export=ALL
set -e
echo "Starting job on worker node"
/fusion/usc/opt/python/2.7.11/bin/python2.7 /home/%u/PyFusionGUI/PyFusionGUI/SLURM/temp.py
