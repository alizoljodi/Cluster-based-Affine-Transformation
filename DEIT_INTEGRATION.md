# DeiT Model Integration (timm-based)

This document describes the integration of DeiT (Data-efficient Image Transformers) models into the Cluster-based Affine Transformation (CAT) quantization framework using the `timm` library.

## Changes Made

### 1. timm-based DeiT Integration (`hubconf.py`)
- Integrated DeiT models using the `timm` library for reliable, well-tested implementations
- Includes support for:
  - DeiT Tiny (5.7M parameters)
  - DeiT Small (22M parameters) 
  - DeiT Base (86.6M parameters)
  - DeiT Base Distilled (87.3M parameters)
- Uses `timm.create_model()` for consistent model creation
- Automatic pretrained weight loading from timm's model registry
- Graceful error handling when timm is not installed

### 2. Updated Main Script (`main_imagenet.py`)
- Extended argument parser to include DeiT model choices:
  - `deit_tiny_patch16_224`
  - `deit_small_patch16_224`
  - `deit_base_patch16_224`
  - `deit_base_distilled_patch16_224`

## Prerequisites

Before using DeiT models, you need to install the `timm` library:

```bash
pip install timm
```

## Usage Examples

### Command Line Usage
```bash
# Run with DeiT Tiny
python main_imagenet.py --arch deit_tiny_patch16_224 --batch_size 32

# Run with DeiT Base
python main_imagenet.py --arch deit_base_patch16_224 --batch_size 16

# Run with DeiT Base Distilled
python main_imagenet.py --arch deit_base_distilled_patch16_224 --batch_size 16
```

### Batch Script Usage
Use the provided `run_deit_example.sh` script to run all DeiT variants:
```bash
bash run_deit_example.sh
```

## Model Specifications

| Model | Parameters | Embed Dim | Depth | Heads | Patch Size |
|-------|------------|-----------|------|-------|------------|
| DeiT Tiny | 5.7M | 192 | 12 | 3 | 16x16 |
| DeiT Small | 22M | 384 | 12 | 6 | 16x16 |
| DeiT Base | 86.6M | 768 | 12 | 12 | 16x16 |
| DeiT Base Distilled | 87.3M | 768 | 12 | 12 | 16x16 |

## Key Features

1. **timm Integration**: Uses the well-maintained `timm` library for reliable DeiT implementations
2. **Pretrained Weights**: All models support loading pretrained ImageNet weights from timm's registry
3. **Quantization Ready**: Compatible with the existing quantization framework
4. **Distillation Support**: DeiT Base Distilled includes knowledge distillation
5. **Error Handling**: Graceful fallback when timm is not installed
6. **Easy Installation**: Simple pip install for timm dependency

## Notes

- DeiT models require input images of size 224x224
- The distilled model returns averaged predictions during inference
- All models are compatible with the existing CAT quantization pipeline
- Pretrained weights are automatically downloaded from timm's model registry
- If timm is not installed, you'll get a clear error message with installation instructions

## References

- [DeiT Paper](https://arxiv.org/abs/2012.12877): "Training data-efficient image transformers & distillation through attention"
- [Official DeiT Repository](https://github.com/facebookresearch/deit)
- [timm Library](https://github.com/huggingface/pytorch-image-models): PyTorch Image Models with pretrained weights
