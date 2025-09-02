# TorchHub Integration for Transformer Models

This document describes the integration of pretrained transformer models from PyTorch Hub (via `timm`) into the PD-Quant pipeline.

## Overview

The `hubconf.py` file has been updated to support loading pretrained transformer models (ViT, Swin, DeiT) directly from the `timm` library when `pretrained=True` is specified. If `timm` is not available, it falls back to using the local model implementations.

## Supported Models

### Vision Transformer (ViT)
- `vit_small` - ViT-Small (22M parameters)
- `vit_base` - ViT-Base (86M parameters)

### Swin Transformer
- `swin_small` - Swin-Small (49M parameters)
- `swin_base` - Swin-Base (87M parameters)

### Data-efficient image Transformers (DeiT)
- `deit_small` - DeiT-Small (22M parameters)
- `deit_tiny` - DeiT-Tiny (5.9M parameters)

## Usage

### Loading Pretrained Models

```python
from hubconf import vit_small, vit_base, swin_small, swin_base, deit_small, deit_tiny

# Load pretrained models (requires timm to be installed)
model = vit_small(pretrained=True)
model = vit_base(pretrained=True)
model = swin_small(pretrained=True)
model = swin_base(pretrained=True)
model = deit_small(pretrained=True)
model = deit_tiny(pretrained=True)

# Load local implementations (no external dependencies)
model = vit_small(pretrained=False)
model = vit_base(pretrained=False)
# ... etc
```

### Installation

To use pretrained models, install `timm`:

```bash
pip install timm
```

### Integration with PD-Quant

The models can be used directly with the PD-Quant pipeline:

```python
# Load pretrained model
model = vit_small(pretrained=True)

# Apply PFQ quantization
from quant.pfq_wrappers import replace_with_pfq_wrappers
model, replaced = replace_with_pfq_wrappers(model, w_bit=2, a_bit=2)

# Use in calibration
from ptq.pfq_pd_calibrator import PFQ_PD_Calibrator
calibrator = PFQ_PD_Calibrator(model, w_bit=2, a_bit=2)
# ... calibration process
```

## Implementation Details

### Fallback Mechanism

The `hubconf.py` functions implement a fallback mechanism:

1. If `pretrained=True` and `timm` is available:
   - Load pretrained model from `timm`
2. If `pretrained=True` but `timm` is not available:
   - Print warning and use local implementation
3. If `pretrained=False`:
   - Always use local implementation

### Model Compatibility

Both pretrained and local models are compatible with:
- PFQ quantization wrappers
- PD-Quant calibration pipeline
- Standard PyTorch operations

## Testing

### Test Scripts

1. **`test_torchhub_integration.py`** - Tests both pretrained and local model loading
2. **`install_timm_and_test.py`** - Installs timm and tests pretrained models
3. **`test_transformer_models.py`** - Comprehensive testing of model functionality

### Running Tests

```bash
# Test without timm (uses local implementations)
python test_torchhub_integration.py

# Install timm and test pretrained models
python install_timm_and_test.py

# Test transformer model functionality
python test_transformer_models.py
```

## Model Specifications

### Input Requirements
- **Image size**: 224x224 pixels
- **Channels**: 3 (RGB)
- **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Output
- **Classes**: 1000 (ImageNet-1k)
- **Format**: Logits (before softmax)

### Architecture Details

#### ViT Models
- **Patch size**: 16x16
- **Embedding dimension**: 384 (small), 768 (base)
- **Number of layers**: 12 (small), 12 (base)
- **Number of heads**: 6 (small), 12 (base)

#### Swin Models
- **Patch size**: 4x4
- **Window size**: 7x7
- **Embedding dimension**: 96 (small), 128 (base)
- **Number of layers**: 2, 2, 6, 2 (small), 2, 2, 18, 2 (base)

#### DeiT Models
- **Patch size**: 16x16
- **Embedding dimension**: 384 (small), 192 (tiny)
- **Number of layers**: 12 (small), 12 (tiny)
- **Distillation token**: Yes

## Performance Considerations

### Memory Usage
- **ViT-Small**: ~22M parameters, ~88MB GPU memory
- **ViT-Base**: ~86M parameters, ~344MB GPU memory
- **Swin-Small**: ~49M parameters, ~196MB GPU memory
- **Swin-Base**: ~87M parameters, ~348MB GPU memory
- **DeiT-Small**: ~22M parameters, ~88MB GPU memory
- **DeiT-Tiny**: ~5.9M parameters, ~24MB GPU memory

### Inference Speed
- DeiT-Tiny is the fastest for inference
- Swin models are generally faster than ViT models
- Base models are slower but more accurate than small models

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'timm'**
   - Solution: Install timm with `pip install timm`

2. **Model not found in timm**
   - Solution: Check available models with `timm.list_models('vit*')`

3. **CUDA out of memory**
   - Solution: Use smaller models or reduce batch size

4. **Incompatible model architecture**
   - Solution: Use local implementations with `pretrained=False`

### Getting Help

- Check the test scripts for examples
- Verify model compatibility with your PyTorch version
- Ensure sufficient GPU memory for the chosen model

## Future Enhancements

- Support for more transformer architectures
- Integration with other model repositories
- Automatic model downloading and caching
- Performance optimization for specific hardware
