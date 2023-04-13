#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=23:50:00
#SBATCH --job-name=ippo-rware
#SBATCH --array=0-3

algo=$1
env=$2
eps_clip=$3
use_wd=$4
weight_decay=$5

source venv/bin/activate
python3 -q -X faulthandler src/main.py --config=$algo --env-config=gymma with env_args.time_limit=500 env_args.key=$env seed=$SLURM_ARRAY_TASK_ID eps_clip=$eps_clip use_wd=$use_wd weight_decay=$weight_decay
