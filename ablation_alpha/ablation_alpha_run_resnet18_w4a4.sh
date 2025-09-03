#!/bin/bash
#SBATCH -J R18_W4A4_ALPHA_ABLATION
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet18 W4A4 alpha ablation experiment at $(date)"
python ../run_script_seed.py resnet18 --w_bits 4 --a_bits 4 --num_seeds 3 --start_seed 0 --sleep 0.5 --alpha 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --num_clusters 64 --pca_dim 50
echo "Completed ResNet18 W4A4 alpha ablation experiment at $(date)"
