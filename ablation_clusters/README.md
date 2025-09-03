# Cluster Ablation Experiments

This folder contains shell scripts for running cluster ablation experiments across different architectures and bitwidths.

## Cluster Values Tested
All scripts test the following cluster values: `[1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]`

## Fixed Parameters
- **Alpha**: 0.6 (fixed across all experiments)
- **PCA dimensions**: 50 (fixed across all experiments)
- **Number of seeds**: 3
- **Start seed**: 0
- **Sleep between runs**: 0.5 seconds

## Available Scripts

### ResNet18
- `ablation_clusters_run_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `ablation_clusters_run_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `ablation_clusters_run_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `ablation_clusters_run_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

### ResNet50
- `ablation_clusters_run_resnet50_w2a2.sh` - ResNet50 with 2-bit weights and activations
- `ablation_clusters_run_resnet50_w2a4.sh` - ResNet50 with 2-bit weights and 4-bit activations
- `ablation_clusters_run_resnet50_w4a2.sh` - ResNet50 with 4-bit weights and 2-bit activations
- `ablation_clusters_run_resnet50_w4a4.sh` - ResNet50 with 4-bit weights and activations

### MobileNetV2
- `ablation_clusters_run_mobilenetv2_w2a2.sh` - MobileNetV2 with 2-bit weights and activations
- `ablation_clusters_run_mobilenetv2_w2a4.sh` - MobileNetV2 with 2-bit weights and 4-bit activations
- `ablation_clusters_run_mobilenetv2_w4a2.sh` - MobileNetV2 with 4-bit weights and 2-bit activations
- `ablation_clusters_run_mobilenetv2_w4a4.sh` - MobileNetV2 with 4-bit weights and activations

### RegNetX-600M
- `ablation_clusters_run_regnetx_600m_w2a2.sh` - RegNetX-600M with 2-bit weights and activations
- `ablation_clusters_run_regnetx_600m_w2a4.sh` - RegNetX-600M with 2-bit weights and 4-bit activations
- `ablation_clusters_run_regnetx_600m_w4a2.sh` - RegNetX-600M with 4-bit weights and 2-bit activations
- `ablation_clusters_run_regnetx_600m_w4a4.sh` - RegNetX-600M with 4-bit weights and activations

### RegNetX-3200M
- `ablation_clusters_run_regnetx_3200m_w2a2.sh` - RegNetX-3200M with 2-bit weights and activations
- `ablation_clusters_run_regnetx_3200m_w2a4.sh` - RegNetX-3200M with 2-bit weights and 4-bit activations
- `ablation_clusters_run_regnetx_3200m_w4a2.sh` - RegNetX-3200M with 4-bit weights and 2-bit activations
- `ablation_clusters_run_regnetx_3200m_w4a4.sh` - RegNetX-3200M with 4-bit weights and activations

### MnasNet
- `ablation_clusters_run_mnasnet_w2a2.sh` - MnasNet with 2-bit weights and activations
- `ablation_clusters_run_mnasnet_w2a4.sh` - MnasNet with 2-bit weights and 4-bit activations
- `ablation_clusters_run_mnasnet_w4a2.sh` - MnasNet with 4-bit weights and 2-bit activations
- `ablation_clusters_run_mnasnet_w4a4.sh` - MnasNet with 4-bit weights and activations

## Usage

Each script can be run using SLURM:

```bash
sbatch ablation_clusters_run_resnet18_w2a2.sh
```

Or directly:

```bash
./ablation_clusters_run_resnet18_w2a2.sh
```

## Configuration

All scripts use the following configuration:
- **Number of seeds**: 3
- **Start seed**: 0
- **Sleep between runs**: 0.5 seconds
- **Alpha**: 0.6 (fixed)
- **PCA dimensions**: 50 (fixed)
- **Cluster values**: 1, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256

## Expected Output

Each script will:
1. Run experiments for all 11 cluster values
2. Test each configuration with 3 different seeds
3. Save results to CSV files
4. Provide aggregated statistics across all seeds

## Total Experiments

Each script runs:
- 11 cluster values × 3 seeds = 33 total experiments per script

**Total scripts**: 24 scripts (6 architectures × 4 bitwidth combinations)
**Total experiments**: 24 scripts × 33 experiments = 792 total experiments
