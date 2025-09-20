#!/bin/bash
#SBATCH -J R18_W4A4_minmax_mse
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet18 W4A4 ablation (init_wmode=minmax, init_amode=mse) at $(date)"
python ../run_script_seed.py resnet18 --w_bits 4 --a_bits 4 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.4 --num_clusters 1 --pca_dim 1 --init_wmode minmax --init_amode mse --use_enhanced_cat False
echo "Completed ResNet18 W4A4 ablation (init_wmode=minmax, init_amode=mse) at $(date)"

