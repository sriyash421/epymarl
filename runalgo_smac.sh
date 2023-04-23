#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=43:50:00
#SBATCH --job-name=ippo-smac
#SBATCH --array=0-3

algo=$1
env=$2
eps_clip=$3
use_wd=$4
weight_decay=$5
sampling_dim=$6

source venv/bin/activate
python src/main.py --config=$algo --env-config=sc2 with env_args.map_name=$env seed=$SLURM_ARRAY_TASK_ID eps_clip=$eps_clip use_wd=$use_wd weight_decay=$weight_decay sampling_dim=$sampling_dim
