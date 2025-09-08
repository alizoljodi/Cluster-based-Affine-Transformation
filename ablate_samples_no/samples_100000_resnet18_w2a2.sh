#!/bin/bash
#SBATCH -J R18_W2A2_SAMPLES_100000
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting ResNet18 W2A2 with 100000 samples experiment at $(date)"

# Run experiments for each seed
for seed in 1001 1002 1003; do
    echo "Running experiment with seed=$seed"
    python ../main_imagenet.py --data_path /home/alz07xz/imagenet --arch resnet18 \
        --n_bits_w 2 --n_bits_a 2 --weight 0.01 --T 4.0 --lamb_c 0.02 --seed $seed \
        --alpha 0.5 --num_clusters 64 --pca_dim 50 --num_samples 100000
    sleep 0.5
done

echo "Completed ResNet18 W2A2 with 100000 samples experiment at $(date)"
