# Prepositive Feature Quantization (PFQ) + Feature-Loss-Aware Optimization (FLAO)

This repository now includes an implementation of **Prepositive Feature Quantization (PFQ)** with **Feature-Loss-Aware Optimization (FLAO)** for the PD-Quant pipeline. This implementation integrates prepositive activation quantization (LSQ on layer inputs) and a feature-first then weight-last calibration schedule.

## Overview

PFQ+FLAO provides a novel approach to model quantization that:

1. **Prepositive Quantization**: Applies LSQ activation quantization to layer inputs (prepositive)
2. **Feature-Loss-Aware Optimization**: Uses a two-stage calibration schedule:
   - **Stage A (Feature-first)**: Optimize LSQ step sizes while keeping weights FP
   - **Stage B (Weight-last)**: Freeze LSQ and optimize AdaRound for weights
3. **PD Loss**: Uses prediction-difference loss at logits against a frozen full-precision teacher
4. **Layer Isolation**: Calibrates each layer individually while keeping others FP

## Key Features

- ✅ **LSQActivation**: Learned step size quantization for activations
- ✅ **AdaRoundWeight**: Adaptive rounding for weight quantization  
- ✅ **FLAO Schedule**: Feature-first, weight-last optimization
- ✅ **PD Loss**: KL divergence loss at logits
- ✅ **QDrop Regularization**: Optional dropout for activation quantization
- ✅ **Layer Isolation**: Individual layer calibration
- ✅ **BatchNorm Freezing**: Keeps BN layers in eval mode
- ✅ **First/Last Layer Protection**: Optional 8-bit quantization for first/last layers

## Installation

The PFQ components are integrated into the existing codebase. No additional installation is required.

## Usage

### Basic Usage

Enable PFQ by adding the `--use_pfq` flag to your existing command:

```bash
python main_imagenet.py \
    --data_path /path/to/imagenet \
    --arch resnet18 \
    --n_bits_w 2 \
    --n_bits_a 2 \
    --use_pfq \
    --feature_steps 1000 \
    --weight_steps 4000 \
    --lr_feature 4e-5 \
    --lr_weight 1e-3 \
    --pfq_T 1.0 \
    --pfq_lam 1.0 \
    --qdrop_p 0.1 \
    --num_samples 1024 \
    --batch_size 32 \
    --seed 42
```

### New Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_pfq` | False | Enable PFQ+FLAO calibration |
| `--feature_steps` | 1000 | Number of steps for feature optimization (Stage A) |
| `--weight_steps` | 4000 | Number of steps for weight optimization (Stage B) |
| `--lr_feature` | 4e-5 | Learning rate for feature optimization |
| `--lr_weight` | 1e-3 | Learning rate for weight optimization |
| `--pfq_T` | 1.0 | Temperature for PD loss |
| `--pfq_lam` | 1.0 | Lambda for PD loss |
| `--qdrop_p` | 0.0 | QDrop probability for feature optimization |
| `--skip_first_last` | True | Skip first and last layers (keep 8-bit) |

### Hyperparameter Recommendations

#### For W2A2 Quantization
```bash
--feature_steps 2000 \
--weight_steps 8000 \
--lr_feature 4e-5 \
--lr_weight 1e-3 \
--qdrop_p 0.1
```

#### For W4A4 Quantization
```bash
--feature_steps 1000 \
--weight_steps 4000 \
--lr_feature 4e-5 \
--lr_weight 1e-3 \
--qdrop_p 0.0
```

#### Conservative Settings (Stable)
```bash
--feature_steps 5000 \
--weight_steps 20000 \
--lr_feature 2e-5 \
--lr_weight 5e-4 \
--qdrop_p 0.2
```

## Implementation Details

### Core Components

1. **`quant/pfq_quantizers.py`**
   - `LSQActivation`: Learned step size quantization for activations
   - `AdaRoundWeight`: Adaptive rounding for weight quantization

2. **`quant/pfq_wrappers.py`**
   - `QuantizedConv2dWrapper`: Wrapper for Conv2d layers
   - `QuantizedLinearWrapper`: Wrapper for Linear layers
   - `replace_with_pfq_wrappers()`: Model transformation function
   - `set_wrappers_mode()`: Runtime quantization control

3. **`quant/pfq_pd_calibrator.py`**
   - `PFQ_PD_Calibrator`: Main calibration class
   - `pd_loss_from_logits()`: PD loss implementation
   - `flao_pd_layer()`: FLAO calibration for single layer
   - `calibrate_all_layers()`: Full model calibration

### Calibration Process

1. **Model Preparation**
   - Replace Conv2d/Linear layers with PFQ wrappers
   - Freeze BatchNorm layers
   - Set first/last layers to 8-bit (optional)

2. **Layer-by-Layer Calibration**
   - **Stage A**: Optimize LSQ step sizes
     - Current layer: act-quant (with QDrop), weight-FP
     - Other layers: FP
     - Loss: PD loss at logits
   
   - **Stage B**: Optimize AdaRound weights
     - Current layer: act-quant, weight-quant
     - Other layers: FP
     - Loss: PD loss at logits

3. **Finalization**
   - Copy quantized weights to original layers
   - Return finalized quantized model

### Loss Function

The PD loss uses KL divergence between logits:

```python
def pd_loss_from_logits(logits_q, logits_fp, T=1.0, lam=1.0):
    return F.kl_div(
        F.log_softmax(logits_q / T, dim=1),
        F.softmax(logits_fp / T, dim=1),
        reduction="batchmean"
    ) / lam
```

## Testing

Run the test script to verify the implementation:

```bash
python test_pfq.py
```

This will test:
- LSQActivation quantizer
- AdaRoundWeight quantizer
- PFQ wrappers
- PD loss function
- Model replacement
- Calibrator initialization

## Examples

See `example_pfq_usage.py` for detailed usage examples and comparisons.

### Comparison with Traditional Quantization

| Aspect | Traditional | PFQ+FLAO |
|--------|-------------|----------|
| Quantization Order | Simultaneous | Sequential (layer-by-layer) |
| Loss Function | Reconstruction | PD (KL divergence) |
| Optimization | Joint | Feature-first, weight-last |
| Regularization | None | QDrop (optional) |
| Layer Isolation | No | Yes |
| Calibration Schedule | Single stage | Two-stage (FLAO) |

## Performance Expectations

PFQ+FLAO is designed to provide better accuracy at low bitwidths (especially W2A2) compared to traditional quantization methods. Key benefits:

- **Better W2A2 Performance**: Improved accuracy at 2-bit quantization
- **Stable Training**: QDrop regularization helps with training stability
- **Layer Isolation**: Individual layer calibration reduces interference
- **PD Loss**: Direct optimization of prediction difference

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or number of calibration samples
2. **Training Instability**: Increase QDrop probability or reduce learning rates
3. **Poor Accuracy**: Try conservative hyperparameters or increase calibration steps
4. **Slow Training**: Reduce feature_steps and weight_steps for faster training

### Debug Tips

- Start with W4A4 before trying W2A2
- Use small datasets for initial testing
- Monitor loss curves during calibration
- Check that BatchNorm layers are frozen

## Citation

If you use this implementation in your research, please cite the original PFQ paper and this implementation.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This implementation follows the same license as the original codebase.
