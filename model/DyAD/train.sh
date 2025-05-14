#!/bin/bash

#SBATCH --job-name=pruning
#SBATCH --nodelist=ariel-v6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --partition batch_ugrad
#SBATCH --time 1-0
#SBATCH -o compression/slurm-%A-%x.out
#SBATCH -e compression/slurm-%A-%x.err



echo "Job started at: $(date)"

python mainscript.py --config_path model_params.json --fold_num 0

echo "Job ended at: $(date)"


# letting slurm know this code finished without any problem
exit 0


