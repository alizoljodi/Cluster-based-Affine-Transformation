# Alpha Ablation Experiments

This folder contains shell scripts for running alpha ablation experiments across different architectures and bitwidths.

## Alpha Values Tested
All scripts test the following alpha values: `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`

## Available Scripts

### ResNet18
- `ablation_alpha_run_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `ablation_alpha_run_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `ablation_alpha_run_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `ablation_alpha_run_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

### ResNet50
- `ablation_alpha_run_resnet50_w2a2.sh` - ResNet50 with 2-bit weights and activations
- `ablation_alpha_run_resnet50_w2a4.sh` - ResNet50 with 2-bit weights and 4-bit activations
- `ablation_alpha_run_resnet50_w4a2.sh` - ResNet50 with 4-bit weights and 2-bit activations
- `ablation_alpha_run_resnet50_w4a4.sh` - ResNet50 with 4-bit weights and activations

### MobileNetV2
- `ablation_alpha_run_mobilenetv2_w2a2.sh` - MobileNetV2 with 2-bit weights and activations
- `ablation_alpha_run_mobilenetv2_w2a4.sh` - MobileNetV2 with 2-bit weights and 4-bit activations
- `ablation_alpha_run_mobilenetv2_w4a2.sh` - MobileNetV2 with 4-bit weights and 2-bit activations
- `ablation_alpha_run_mobilenetv2_w4a4.sh` - MobileNetV2 with 4-bit weights and activations

### RegNetX-600M
- `ablation_alpha_run_regnetx_600m_w2a2.sh` - RegNetX-600M with 2-bit weights and activations
- `ablation_alpha_run_regnetx_600m_w2a4.sh` - RegNetX-600M with 2-bit weights and 4-bit activations
- `ablation_alpha_run_regnetx_600m_w4a2.sh` - RegNetX-600M with 4-bit weights and 2-bit activations
- `ablation_alpha_run_regnetx_600m_w4a4.sh` - RegNetX-600M with 4-bit weights and activations

### RegNetX-3200M
- `ablation_alpha_run_regnetx_3200m_w2a2.sh` - RegNetX-3200M with 2-bit weights and activations
- `ablation_alpha_run_regnetx_3200m_w2a4.sh` - RegNetX-3200M with 2-bit weights and 4-bit activations
- `ablation_alpha_run_regnetx_3200m_w4a2.sh` - RegNetX-3200M with 4-bit weights and 2-bit activations
- `ablation_alpha_run_regnetx_3200m_w4a4.sh` - RegNetX-3200M with 4-bit weights and activations

### MnasNet
- `ablation_alpha_run_mnasnet_w2a2.sh` - MnasNet with 2-bit weights and activations
- `ablation_alpha_run_mnasnet_w2a4.sh` - MnasNet with 2-bit weights and 4-bit activations
- `ablation_alpha_run_mnasnet_w4a2.sh` - MnasNet with 4-bit weights and 2-bit activations
- `ablation_alpha_run_mnasnet_w4a4.sh` - MnasNet with 4-bit weights and activations

## Usage

Each script can be run using SLURM:

```bash
sbatch ablation_alpha_run_resnet18_w2a2.sh
```

Or directly:

```bash
./ablation_alpha_run_resnet18_w2a2.sh
```

## Configuration

All scripts use the following configuration:
- **Number of seeds**: 3
- **Start seed**: 0
- **Sleep between runs**: 0.5 seconds
- **Number of clusters**: 64
- **PCA dimensions**: 50
- **Alpha values**: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

## Expected Output

Each script will:
1. Run experiments for all 10 alpha values
2. Test each configuration with 3 different seeds
3. Save results to CSV files
4. Provide aggregated statistics across all seeds

## Total Experiments

Each script runs:
- 10 alpha values × 3 seeds = 30 total experiments per script

**Total scripts**: 24 scripts (6 architectures × 4 bitwidth combinations)
**Total experiments**: 24 scripts × 30 experiments = 720 total experiments
