#!/bin/bash

#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=23:50:00
#SBATCH -o ${SCRATCH}/ippo_logs/%j.out
#SBATCH --job-name=ippo
#SBATCH --array=0-4

algo=$1
env=$2
eps_clip=$3
use_wd=$4
weight_decay=$5

source $HOME/mappo-plus/env/bin/activate
python src/main.py --config=$algo --env-config=sc2 with env_args.map_name=$env seed=$SLURM_ARRAY_TASK_ID eps_clip=$eps_clip use_wd=$use_wd weight_decay=$weight_decay