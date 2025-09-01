#!/bin/bash
#SBATCH -J R600_W2A2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
python run_script_seed.py regnetx_600m --w_bits 2 --a_bits 2 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.4 --num_clusters 1 --pca_dim 1