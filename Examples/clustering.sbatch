#!/bin/bash
## Specify the (p)artition or queue to be placed in.
#SBATCH -p short
## Specify the number of CPUS to utilize
#SBATCH -n 2
## Specify the number of nodes/CPUS to use. SET TO 1!
#SBATCH -N 1
## Estimate how long it will take your script to run
#SBATCH -t 8
## Limit the amount of ram per CPU core
#SBATCH --mem 115G
## Specify destinataion for program output (NOT THE SAVED ANALYSIS OBJECT)
#SBATCH -o PyFusion-%j.out
## Export your environment variables to the job
#SBATCH --export=ALL
## Set the script to FAILED status if there is any error
set -e
## Run the script
/fusion/usc/opt/python/2.7.11/bin/python2.7 run_clustering.py
