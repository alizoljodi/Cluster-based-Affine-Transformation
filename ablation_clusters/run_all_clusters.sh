#!/bin/bash

# Script to submit all cluster ablation experiments to SLURM
# This script will submit all 24 cluster ablation scripts with appropriate output file names

echo "=========================================="
echo "SUBMITTING ALL CLUSTER ABLATION EXPERIMENTS"
echo "=========================================="
echo "Total scripts to submit: 24"
echo "Total experiments: 792 (24 scripts × 33 experiments each)"
echo "=========================================="

# ResNet18 experiments
echo "Submitting ResNet18 experiments..."
sbatch --output=./logs/resnet18_w2a2_clusters.out ablation_clusters_run_resnet18_w2a2.sh
sbatch --output=./logs/resnet18_w2a4_clusters.out ablation_clusters_run_resnet18_w2a4.sh
sbatch --output=./logs/resnet18_w4a2_clusters.out ablation_clusters_run_resnet18_w4a2.sh
sbatch --output=./logs/resnet18_w4a4_clusters.out ablation_clusters_run_resnet18_w4a4.sh

# ResNet50 experiments
echo "Submitting ResNet50 experiments..."
sbatch --output=./logs/resnet50_w2a2_clusters.out ablation_clusters_run_resnet50_w2a2.sh
sbatch --output=./logs/resnet50_w2a4_clusters.out ablation_clusters_run_resnet50_w2a4.sh
sbatch --output=./logs/resnet50_w4a2_clusters.out ablation_clusters_run_resnet50_w4a2.sh
sbatch --output=./logs/resnet50_w4a4_clusters.out ablation_clusters_run_resnet50_w4a4.sh

# MobileNetV2 experiments
echo "Submitting MobileNetV2 experiments..."
sbatch --output=./logs/mobilenetv2_w2a2_clusters.out ablation_clusters_run_mobilenetv2_w2a2.sh
sbatch --output=./logs/mobilenetv2_w2a4_clusters.out ablation_clusters_run_mobilenetv2_w2a4.sh
sbatch --output=./logs/mobilenetv2_w4a2_clusters.out ablation_clusters_run_mobilenetv2_w4a2.sh
sbatch --output=./logs/mobilenetv2_w4a4_clusters.out ablation_clusters_run_mobilenetv2_w4a4.sh

# RegNetX-600M experiments
echo "Submitting RegNetX-600M experiments..."
sbatch --output=./logs/regnetx_600m_w2a2_clusters.out ablation_clusters_run_regnetx_600m_w2a2.sh
sbatch --output=./logs/regnetx_600m_w2a4_clusters.out ablation_clusters_run_regnetx_600m_w2a4.sh
sbatch --output=./logs/regnetx_600m_w4a2_clusters.out ablation_clusters_run_regnetx_600m_w4a2.sh
sbatch --output=./logs/regnetx_600m_w4a4_clusters.out ablation_clusters_run_regnetx_600m_w4a4.sh

# RegNetX-3200M experiments
echo "Submitting RegNetX-3200M experiments..."
sbatch --output=./logs/regnetx_3200m_w2a2_clusters.out ablation_clusters_run_regnetx_3200m_w2a2.sh
sbatch --output=./logs/regnetx_3200m_w2a4_clusters.out ablation_clusters_run_regnetx_3200m_w2a4.sh
sbatch --output=./logs/regnetx_3200m_w4a2_clusters.out ablation_clusters_run_regnetx_3200m_w4a2.sh
sbatch --output=./logs/regnetx_3200m_w4a4_clusters.out ablation_clusters_run_regnetx_3200m_w4a4.sh

# MnasNet experiments
echo "Submitting MnasNet experiments..."
sbatch --output=./logs/mnasnet_w2a2_clusters.out ablation_clusters_run_mnasnet_w2a2.sh
sbatch --output=./logs/mnasnet_w2a4_clusters.out ablation_clusters_run_mnasnet_w2a4.sh
sbatch --output=./logs/mnasnet_w4a2_clusters.out ablation_clusters_run_mnasnet_w4a2.sh
sbatch --output=./logs/mnasnet_w4a4_clusters.out ablation_clusters_run_mnasnet_w4a4.sh

echo "=========================================="
echo "ALL EXPERIMENTS SUBMITTED!"
echo "=========================================="
echo "Check job status with: squeue -u $USER"
echo "Monitor output files: ls -la ./logs/*clusters.out"
echo "=========================================="
