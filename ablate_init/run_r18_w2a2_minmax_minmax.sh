#!/bin/bash
#SBATCH -J R18_W2A2_minmax_minmax
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet18 W2A2 ablation (init_wmode=minmax, init_amode=minmax) at $(date)"
python ../run_script_seed.py resnet18 --w_bits 2 --a_bits 2 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.4 --num_clusters 1 --pca_dim 1 --init_wmode minmax --init_amode minmax
echo "Completed ResNet18 W2A2 ablation (init_wmode=minmax, init_amode=minmax) at $(date)"

