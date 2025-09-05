#!/bin/bash
#SBATCH -J REG600M_W4A4_PCA_ABLATION
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting RegNetX-600M W4A4 PCA ablation experiment at $(date)"
python ../run_script_seed.py regnetx_600m --w_bits 4 --a_bits 4 --num_seeds 3 --start_seed 1000 --sleep 0.5 --alpha 0.6 --num_clusters 64 --pca_dim 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220
echo "Completed RegNetX-600M W4A4 PCA ablation experiment at $(date)"
