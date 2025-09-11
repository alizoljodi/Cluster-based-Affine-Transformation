# Samples Ablation Experiments

This folder contains shell scripts for running samples ablation experiments on ResNet18 architecture. The experiments test how the number of calibration samples affects the performance of the Cluster-based Affine Transformation (CAT) method.

## Sample Values Tested

All scripts test the following sample counts: `[10, 50, 100, 500, 1000, 5000, 10000, 100000, 200000, 500000, 1000000]`

## Available Scripts

### Individual Sample Count Scripts

Each sample count has 4 scripts corresponding to different bitwidth combinations:

#### 10 Samples
- `samples_10_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_10_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_10_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_10_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 50 Samples
- `samples_50_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_50_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_50_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_50_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 100 Samples
- `samples_100_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_100_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_100_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_100_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 500 Samples
- `samples_500_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_500_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_500_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_500_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 1000 Samples
- `samples_1000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_1000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_1000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_1000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 5000 Samples
- `samples_5000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_5000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_5000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_5000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 10000 Samples
- `samples_10000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_10000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_10000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_10000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 100000 Samples
- `samples_100000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_100000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_100000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_100000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 200000 Samples
- `samples_200000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_200000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_200000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_200000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 500000 Samples
- `samples_500000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_500000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_500000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_500000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

#### 1000000 Samples
- `samples_1000000_resnet18_w2a2.sh` - ResNet18 with 2-bit weights and activations
- `samples_1000000_resnet18_w2a4.sh` - ResNet18 with 2-bit weights and 4-bit activations
- `samples_1000000_resnet18_w4a2.sh` - ResNet18 with 4-bit weights and 2-bit activations
- `samples_1000000_resnet18_w4a4.sh` - ResNet18 with 4-bit weights and activations

## Usage

### Running Individual Experiments

Each script can be run using SLURM:

```bash
sbatch samples_1000_resnet18_w2a2.sh
```

Or directly:

```bash
./samples_1000_resnet18_w2a2.sh
```

### Running All Experiments

To run all sample ablation experiments at once:

```bash
./run_all_samples.sh
```

This will submit all 44 scripts (11 sample counts × 4 bitwidth combinations) to the SLURM queue, with each script running 3 seeds for a total of 132 experiments.

## Configuration

All scripts use the following configuration:
- **Architecture**: ResNet18
- **Seeds**: 1001, 1002, 1003 (each script runs 3 experiments with different seeds)
- **Alpha**: 0.5
- **Number of clusters**: 64
- **PCA dimensions**: 50
- **Data path**: /home/alz07xz/imagenet
- **Sample counts**: 10, 50, 100, 500, 1000, 5000, 10000, 100000, 200000, 500000, 1000000

## Expected Output

Each script will:
1. Run 3 experiments with the specified number of calibration samples (seeds: 1001, 1002, 1003)
2. Test the CAT method with the given bitwidth configuration
3. Save results to CSV files for each seed
4. Provide performance metrics (Top-1 and Top-5 accuracy) averaged across seeds

## Total Experiments

- **Sample counts**: 11 different values
- **Bitwidth combinations**: 4 different configurations per sample count
- **Seeds**: 3 different seeds per configuration
- **Total experiments**: 11 × 4 × 3 = 132 experiments

## Monitoring Progress

Check the status of submitted jobs:

```bash
squeue -u $USER
```

View job output:

```bash
cat slurm-<job_id>.out
```

## Expected Results

The experiments will help determine:
1. How the number of calibration samples affects CAT performance
2. Whether there's a diminishing returns effect with more samples
3. The optimal number of samples for different bitwidth configurations
4. The trade-off between calibration time and accuracy improvement
