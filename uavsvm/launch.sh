#!/bin/bash -l
#SBATCH -p ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --partition=a100
#SBATCH -c 48
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhj1g23@soton.ac.uk
#SBATCH --time=3:00:00


python uavdnn.py