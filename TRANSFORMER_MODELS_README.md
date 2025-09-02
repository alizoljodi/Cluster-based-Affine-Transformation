# Transformer Models Support

This repository now supports Vision Transformer (ViT), Swin Transformer, and DeiT models for quantization experiments.

## Supported Models

### Vision Transformer (ViT)
- **ViT-Small**: 384 embedding dim, 12 layers, 6 heads
- **ViT-Base**: 768 embedding dim, 12 layers, 12 heads
- **ViT-Large**: 1024 embedding dim, 24 layers, 16 heads

### Swin Transformer
- **Swin-Small**: 96 embedding dim, [2,2,18,2] depths, [3,6,12,24] heads
- **Swin-Base**: 128 embedding dim, [2,2,18,2] depths, [4,8,16,32] heads
- **Swin-Large**: 192 embedding dim, [2,2,18,2] depths, [6,12,24,48] heads

### DeiT (Data-efficient image Transformers)
- **DeiT-Tiny**: 192 embedding dim, 12 layers, 3 heads
- **DeiT-Small**: 384 embedding dim, 12 layers, 6 heads
- **DeiT-Base**: 768 embedding dim, 12 layers, 12 heads

## Usage

### Basic Usage

```bash
# Run ViT-Small experiment
python main_imagenet.py \
    --data_path /path/to/imagenet \
    --arch vit_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size 32 \
    --num_samples 1024

# Run Swin-Small experiment
python main_imagenet.py \
    --data_path /path/to/imagenet \
    --arch swin_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size 32 \
    --num_samples 1024

# Run DeiT-Small experiment
python main_imagenet.py \
    --data_path /path/to/imagenet \
    --arch deit_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size 32 \
    --num_samples 1024
```

### With PFQ (Prepositive Feature Quantization)

```bash
# Run ViT-Small with PFQ
python main_imagenet.py \
    --data_path /path/to/imagenet \
    --arch vit_small \
    --n_bits_w 2 \
    --n_bits_a 2 \
    --use_pfq \
    --feature_steps 1000 \
    --weight_steps 4000 \
    --lr_feature 4e-5 \
    --lr_weight 1e-3 \
    --qdrop_p 0.1
```

## Model Architectures

### Vision Transformer (ViT)
- **Patch Embedding**: Convolutional patch embedding
- **Position Embedding**: Learnable position embeddings
- **Transformer Blocks**: Multi-head self-attention + MLP
- **Classification Head**: Linear classifier on [CLS] token

### Swin Transformer
- **Patch Embedding**: Convolutional patch embedding
- **Window Attention**: Local window-based self-attention
- **Shifted Windows**: Alternating window shifts for cross-window connections
- **Patch Merging**: Downsampling between stages
- **Hierarchical Structure**: Multi-stage architecture

### DeiT
- **Distillation Token**: Additional learnable token for knowledge distillation
- **Teacher-Student Training**: Distillation from a larger teacher model
- **Same Architecture as ViT**: But with distillation capabilities

## Model Specifications

| Model | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|-------|------------|-----------|---------------|
| ViT-Small | 22.1 | 4.6 | 81.2 |
| ViT-Base | 86.6 | 17.6 | 84.5 |
| Swin-Small | 49.6 | 8.7 | 83.2 |
| Swin-Base | 87.8 | 15.4 | 85.2 |
| DeiT-Tiny | 5.7 | 1.3 | 72.2 |
| DeiT-Small | 22.1 | 4.6 | 79.8 |

*Note: Accuracy numbers are approximate and may vary based on training setup.*

## Experimental Scripts

### Quick Test
```bash
python run_transformer_detailed.py /path/to/imagenet --quick
```

### Full Experiments
```bash
# Standard experiments
python run_transformer_detailed.py /path/to/imagenet

# With PFQ
python run_transformer_detailed.py /path/to/imagenet --pfq
```

### Shell Script
```bash
# Update DATA_PATH in the script first
bash run_transformer_experiments.sh
```

## Model Files

- `models/vit.py`: Vision Transformer implementation
- `models/swin.py`: Swin Transformer implementation  
- `models/deit.py`: DeiT implementation
- `hubconf.py`: Updated with new model functions
- `main_imagenet.py`: Updated with new model choices

## Key Features

### For ViT Models
- ✅ Patch embedding with configurable patch size
- ✅ Learnable position embeddings
- ✅ Multi-head self-attention
- ✅ MLP blocks with GELU activation
- ✅ Layer normalization
- ✅ Classification head

### For Swin Models
- ✅ Window-based attention
- ✅ Shifted window mechanism
- ✅ Patch merging for downsampling
- ✅ Hierarchical feature extraction
- ✅ Relative position bias

### For DeiT Models
- ✅ Distillation token
- ✅ Teacher-student training support
- ✅ Same architecture as ViT
- ✅ Knowledge distillation capabilities

## Quantization Considerations

### Transformer-Specific Challenges
1. **Attention Weights**: Large attention matrices may need special quantization strategies
2. **Position Embeddings**: May need different quantization than other weights
3. **Layer Norm**: Often kept in full precision for stability
4. **Activation Ranges**: May vary significantly between layers

### Recommended Settings
- **W4A4**: Good balance of accuracy and efficiency
- **W2A2**: Aggressive quantization, may need PFQ for good results
- **First/Last Layers**: Consider keeping at 8-bit for better accuracy

## Performance Tips

1. **Batch Size**: Start with smaller batch sizes (16-32) for transformer models
2. **Memory**: Transformer models may require more GPU memory
3. **Calibration**: Use more calibration samples (1024-2048) for better results
4. **PFQ**: Especially beneficial for low-bit quantization (W2A2)

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Use mixed precision training if available
3. **Poor Accuracy**: Try different quantization bitwidths or use PFQ
4. **Convergence Issues**: Adjust learning rates or use warmup

### Model-Specific Notes
- **ViT**: May need longer training for good quantization results
- **Swin**: Window attention may be more sensitive to quantization
- **DeiT**: Distillation token may need special handling during quantization

## Citation

If you use these transformer models in your research, please cite the original papers:

```bibtex
@inproceedings{dosovitskiy2021vit,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  booktitle={ICLR},
  year={2021}
}

@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={ICCV},
  year={2021}
}

@inproceedings{touvron2021deit,
  title={Training data-efficient image transformers & distillation through attention},
  author={Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and J{\'e}gou, Herv{\'e}},
  booktitle={ICML},
  year={2021}
}
```
