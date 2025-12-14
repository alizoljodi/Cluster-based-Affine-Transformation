#!/bin/bash
#SBATCH -J R18_W2A4_mse_mse
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet18 W2A4 ablation (init_wmode=mse, init_amode=mse) at $(date)"
python ../run_script_seed.py resnet18 --w_bits 2 --a_bits 4 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.5 --num_clusters 56 --pca_dim 40 --init_wmode mse --init_amode mse
echo "Completed ResNet18 W2A4 ablation (init_wmode=mse, init_amode=mse) at $(date)"



