#!/bin/bash
#SBATCH -J R600_W4A4
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting RegNetX-600M W4A4 experiment at $(date)"
python run_script_seed.py regnetx_600m --w_bits 4 --a_bits 4 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.3 --num_clusters 136 --pca_dim 50
echo "Completed RegNetX-600M W4A4 experiment at $(date)"