#!/bin/bash
#SBATCH -J MBV2_W4A2
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00


source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting MobileNetV2 W4A2 experiment at $(date)"
python run_script_seed.py mobilenetv2 --w_bits 4 --a_bits 2 --num_seeds 10 --start_seed 0 --sleep 0.5 --alpha 0.2 --num_clusters 1 --pca_dim 1
echo "Completed MobileNetV2 W4A2 experiment at $(date)"