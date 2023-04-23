
# sbatch runalgo_rware.sh ippo_ns_rware rware:rware-tiny-2ag-v1 0.2 False 0.0
# sbatch runalgo_rware.sh ippo_ns_rware rware:rware-tiny-2ag-v1 0.5 False 0.0
# sbatch runalgo_rware.sh ippo_ns_rware rware:rware-tiny-2ag-v1 0.2 True 0.1
# sbatch runalgo_rware.sh ippo_ns_rware rware:rware-tiny-2ag-v1 0.2 True 0.01

sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-small-4ag-v1 0.2 False 0.0 8
sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-tiny-4ag-v1 0.2 False 0.0 8
sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-tiny-2ag-v1 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac 2s3z 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac corridor 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac MMM2 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac 6h_vs_8z 0.2 False 0.0 8
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z_vs_3s6z 0.2 False 0.0 8

sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-small-4ag-v1 0.2 False 0.0 32
sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-tiny-4ag-v1 0.2 False 0.0 32
sbatch runalgo_rware.sh ippo_t_ns_rware rware:rware-tiny-2ag-v1 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac 2s3z 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac corridor 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac MMM2 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac 6h_vs_8z 0.2 False 0.0 32
sbatch runalgo_smac.sh ippo_t_ns_smac 3s5z_vs_3s6z 0.2 False 0.0 32

# sbatch runalgo_smac.sh ippo_ns_smac MMM2 0.2 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac MMM2 0.5 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac MMM2 0.2 True 0.001

# sbatch runalgo_smac.sh ippo_ns_smac 6h_vs_8z 0.2 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 6h_vs_8z 0.5 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 6h_vs_8z 0.2 True 0.001

#  sbatch runalgo_smac.sh ippo_ns_smac 5m_vs_6m 0.2 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 5m_vs_6m 0.5 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 5m_vs_6m 0.2 True 0.001


#  sbatch runalgo_smac.sh ippo_ns_smac 3s5z_vs_3s6z 0.2 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 3s5z_vs_3s6z 0.5 False 0.0
#  sbatch runalgo_smac.sh ippo_ns_smac 3s5z_vs_3s6z 0.2 True 0.001
