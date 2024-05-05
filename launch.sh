#!/bin/bash -l
#SBATCH -p ecsstaff
#SBATCH --account=ecsstaff
#SBATCH --partition=a100
#SBATCH -c 48
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jhj1g23@soton.ac.uk
#SBATCH --time=06:00:00
source /local/software/conda/miniconda-py3-new/etc/profile.d/conda.sh
conda activate challenge
python run_ac_offline.py --seed 2 --env_name Walker2d --dataset medrep --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000

