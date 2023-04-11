#!/bin/bash

algo=$1
env=$2
eps_clip=0.2
use_wd=False
weight_decay=0.1

for i in {0..2}
do
   python src/main.py --config=$algo --env-config=sc2 with env_args.map_name=$env seed=$i eps_clip=$eps_clip use_wd=$use_wd weight_decay=$weight_decay &
   echo "Running with $algo and $env for seed=$i eps_clip=$eps_clip use_wd=$use_wd weight_decay=$weight_decay"
   sleep 2s
done

python3 src/main.py --config=test_configs/${algo}_${env_acr} --env-config=${env_config} with env_args.time_limit=${time_limit} env_args.key=${env}