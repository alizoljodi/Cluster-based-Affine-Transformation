<div align="center">
<h2>CAT</h2>
<h4>CAT: POST-TRAINING QUANTIZATION ERROR REDUCTION VIA CLUSTER-BASED AFFINE TRANSFORMATION</h4>
</div>

## Overview
CAT (Cluster-based Affine Transformation) is a post-training quantization error reduction method that improves the accuracy of quantized neural networks by applying cluster-specific affine transformations to the model outputs.

## Features
- **Cluster-based Analysis**: Groups similar quantization errors using K-means clustering
- **Affine Transformation**: Learns per-cluster gamma and beta parameters for error correction
- **PCA Dimensionality Reduction**: Optional PCA for efficient clustering of high-dimensional logits
- **Alpha Blending**: Configurable blending between original and corrected outputs
- **Multi-seed Evaluation**: Comprehensive evaluation across multiple random seeds

## Installation
```
python >= 3.7.13
numpy >= 1.21.6
torch >= 1.11.0
torchvision >= 0.12.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
matplotlib >= 3.5.0
```

## Usage

### 1. Download pre-trained models
The pre-trained FP models in our experiment come from [BRECQ](https://github.com/yhhhli/BRECQ), they can be downloaded from [here](https://github.com/yhhhli/BRECQ/releases/tag/v1.0).
Modify the path of the pre-trained model in ```hubconf.py```.

### 2. Run single experiment
```bash
python main_imagenet.py --arch resnet18 --n_bits_w 4 --n_bits_a 4 --seed 42
```

### 3. Run multi-seed evaluation
```bash
python run_script_seed.py resnet18 --w_bits 4 --a_bits 4 --num_seeds 10
```

### 4. Run original quantization baseline
```bash
python run_script.py resnet18
```

## Supported Models
- ResNet18, ResNet50
- MobileNetV2, MNasNet
- RegNetX-600M, RegNetX-3200M

## Configuration Options
- `--alpha`: Alpha blending values (default: [0.4])
- `--num_clusters`: Number of clusters for CAT (default: [64])
- `--pca_dim`: PCA dimensions (default: [-1] for no PCA)
- `--num_samples`: Number of samples for logits extraction (default: 1024)

## Output
- Results are automatically saved to CSV files with timestamps
- Comprehensive statistics including mean and standard deviation across seeds
- Optional visualization of cluster-wise logit comparisons

## Package Structure
```
CAT/
├── __init__.py
├── cat.py              # Core CAT implementation
├── get_logits.py       # Logits extraction utilities
└── visualize.py        # Visualization helpers
```

## Reference
This implementation is based on the original PD-Quant framework and extends it with cluster-based affine transformation for quantization error restoration.
