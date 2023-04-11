
sbatch runalgo_rware.sh ippo_ns_rware rware:rware-small-4ag-v1 0.2 False 0.0
sbatch runalgo_rware.sh ippo_ns_rware rware:rware-small-4ag-v1 0.5 False 0.0
sbatch runalgo_rware.sh ippo_ns_rware rware:rware-small-4ag-v1 0.2 True 0.1
sbatch runalgo_rware.sh ippo_ns_rware rware:rware-small-4ag-v1 0.2 True 0.01


sbatch runalgo_smac.sh ippo_ns_smac 2s3z 0.2 False 0.0
sbatch runalgo_smac.sh ippo_ns_smac 2s3z 0.5 False 0.0
sbatch runalgo_smac.sh ippo_ns_smac 2s3z 0.2 True 0.1
sbatch runalgo_smac.sh ippo_ns_smac 2s3z 0.2 True 0.01

sbatch runalgo_smac.sh ippo_ns_smac 3s5z 0.2 False 0.0
sbatch runalgo_smac.sh ippo_ns_smac 3s5z 0.5 False 0.0
sbatch runalgo_smac.sh ippo_ns_smac 3s5z 0.2 True 0.1
sbatch runalgo_smac.sh ippo_ns_smac 3s5z 0.2 True 0.01


