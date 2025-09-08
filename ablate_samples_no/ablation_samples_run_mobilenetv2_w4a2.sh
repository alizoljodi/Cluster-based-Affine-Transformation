#!/bin/bash
#SBATCH -J MBV2_W4A2_SAMPLES_ABLATION
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting MobileNetV2 W4A2 samples ablation experiment at $(date)"

# Define sample values to test
samples_values=(256 512 1024 2048 4096)

# Run experiments for each sample value
for num_samples in "${samples_values[@]}"; do
    echo "Running experiment with num_samples=$num_samples"
    python ../main_imagenet.py --data_path /home/alz07xz/imagenet --arch mobilenetv2 \
        --n_bits_w 4 --n_bits_a 2 --weight 0.1 --T 1.0 --lamb_c 0.005 --seed 1000 \
        --alpha 0.5 --num_clusters 64 --pca_dim 50 --num_samples $num_samples
    sleep 0.5
done

echo "Completed MobileNetV2 W4A2 samples ablation experiment at $(date)"
