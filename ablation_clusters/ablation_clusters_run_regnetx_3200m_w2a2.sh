#!/bin/bash
#SBATCH -J REG3200M_W2A2_CLUSTER_ABLATION
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting RegNetX-3200M W2A2 cluster ablation experiment at $(date)"
python ../run_script_seed.py regnetx_3200m --w_bits 2 --a_bits 2 --num_seeds 3 --start_seed 1000 --sleep 0.5 --alpha 0.6 --num_clusters 1 8 16 24 32 48 64 96 128 192 256 --pca_dim 50
echo "Completed RegNetX-3200M W2A2 cluster ablation experiment at $(date)"
