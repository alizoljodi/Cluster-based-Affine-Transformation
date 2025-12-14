#!/bin/bash
#SBATCH -J R50_W2A2_SPECTRAL
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet50 W2A2 SPECTRAL experiment at $(date)"
python run_script_seed.py resnet50 --w_bits 2 --a_bits 2 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.6 --num_clusters 1 8 16 24 32 48 64 96 128 192 256 --pca_dim 50 --clustering_algorithm spectral
echo "Completed ResNet50 W2A2 SPECTRAL experiment at $(date)"

