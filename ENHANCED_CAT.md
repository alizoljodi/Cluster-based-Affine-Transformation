# Enhanced CAT Implementation

This document describes the Enhanced CAT (Cluster-based Affine Transformation) implementation with advanced optimization techniques.

## Overview

The Enhanced CAT extends the original CAT framework with sophisticated optimization techniques including:

- **Trust Region Updates**: Prevents degradation of PTQ performance
- **EMA Centroid Updates**: Smooth centroid evolution during optimization
- **Cluster Reassignment**: Dynamic cluster assignment based on corrected logits
- **Advanced Loss Functions**: KL divergence, center loss, and separation loss
- **Gradient Clipping**: Stable optimization with gradient norm clipping

## Key Features

### 1. Advanced Loss Functions

```python
# KL Divergence Loss (primary objective)
def kl_to_fp(z_corr, z_fp, T=1.0, eps=1e-8):
    p = torch.softmax(z_fp / T, dim=-1).clamp_min(eps)
    q = torch.softmax(z_corr / T, dim=-1).clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()

# Center Loss (cluster cohesion)
def center_loss(z_corr, y, c):
    return ((z_corr - c[y]).pow(2).sum(dim=1)).mean()

# Separation Loss (cluster separation)
def sep_loss(c, tau=1.0):
    diff = c[:, None, :] - c[None, :, :]
    dist = diff.pow(2).sum(dim=-1).sqrt()
    mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
    return torch.exp(-dist[mask] / tau).mean()
```

### 2. Trust Region Control

The trust region mechanism prevents the optimization from degrading the original PTQ performance:

```python
def trust_region_update(L_ptq, L_ptq_baseline, eps, lam, decay=0.5, min_lam=1e-3):
    if L_ptq.item() > L_ptq_baseline * (1.0 + eps):
        lam = max(min_lam, lam * decay)  # tighten trust region
    return lam
```

### 3. EMA Centroid Updates

Centroids are updated using exponential moving average for smooth evolution:

```python
@torch.no_grad()
def ema_update_centroids(c, z_corr, y, beta_ema=0.9):
    for k in torch.unique(y):
        idx = (y == k).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            c[k] = beta_ema * c[k] + (1 - beta_ema) * z_corr[idx].mean(dim=0)
```

### 4. Dynamic Cluster Reassignment

Clusters are reassigned periodically based on corrected logits:

```python
# k-means assignment on corrected logits
d2 = torch.cdist(z_corr, c)  # [B,K]
y_new = d2.argmin(dim=1)

# Only update if there are significant changes
if (y_new != y).sum().item() > B * 0.05:  # 5% threshold
    y = y_new
```

## Usage

### Command Line Interface

```bash
# Basic enhanced CAT usage
python main_imagenet.py \
    --arch resnet18 \
    --use_enhanced_cat \
    --cat_steps 400 \
    --cat_lr 4e-4 \
    --cat_lam 0.3 \
    --cat_mu 0.3 \
    --alpha 0.4 \
    --num_clusters 64

# Advanced configuration
python main_imagenet.py \
    --arch deit_tiny_patch16_224 \
    --use_enhanced_cat \
    --cat_steps 300 \
    --cat_lr 5e-4 \
    --cat_lam 0.4 \
    --cat_mu 0.2 \
    --cat_tau 1.0 \
    --cat_eps 0.01 \
    --cat_rho 1e-4 \
    --cat_beta_ema 0.95 \
    --cat_reassign_freq 15 \
    --alpha 0.4 \
    --num_clusters 32
```

### Programmatic Usage

```python
from CAT.enhanced_cat import EnhancedCAT

# Initialize enhanced CAT
cat = EnhancedCAT()

# Build cluster-affine model with advanced optimization
cluster_model, gamma_dict, beta_dict, pca, centroids = cat.build_cluster_affine_enhanced(
    all_q, all_fp,
    num_clusters=64,
    num_steps=400,
    lr=4e-4,
    lam=0.3,
    mu=0.3,
    tau=1.0,
    eps=0.01,
    rho=1e-4,
    beta_ema=0.9,
    reassign_freq=10
)

# Evaluate with enhanced correction
top1_acc, top5_acc, _, _, _, _ = cat.evaluate_cluster_affine_enhanced(
    q_model=qnn,
    fp_model=fp_model,
    cluster_model=cluster_model,
    gamma_dict=gamma_dict,
    beta_dict=beta_dict,
    dataloader=test_loader,
    device=device,
    pca=pca,
    alpha=0.4
)
```

## Parameters

### Core Parameters

- `num_clusters`: Number of clusters (default: 64)
- `num_steps`: Optimization steps (default: 400)
- `lr`: Learning rate (default: 4e-4)

### Loss Weights

- `lam`: Center loss weight (default: 0.3)
- `mu`: Separation loss weight (default: 0.3)
- `tau`: Temperature for separation loss (default: 1.0)
- `rho`: Regularization weight (default: 1e-4)

### Trust Region

- `eps`: Trust region epsilon (default: 0.01)
- Controls how much PTQ performance degradation is allowed

### EMA and Reassignment

- `beta_ema`: EMA decay for centroids (default: 0.9)
- `reassign_freq`: Cluster reassignment frequency (default: 10)

## Optimization Process

1. **Initialization**: K-means clustering on quantized logits
2. **Parameter Setup**: Initialize gamma, beta, centroids
3. **Optimization Loop**:
   - Apply affine transformation
   - Compute losses (KL, center, separation, regularization)
   - Backward pass with gradient clipping
   - Trust region update
   - EMA centroid update
   - Periodic cluster reassignment
4. **Evaluation**: Apply learned corrections with alpha blending

## Benefits

1. **Better Convergence**: Advanced optimization techniques lead to better solutions
2. **Stability**: Trust region prevents performance degradation
3. **Adaptability**: Dynamic cluster reassignment adapts to optimization progress
4. **Robustness**: EMA updates provide smooth centroid evolution
5. **Flexibility**: Extensive parameter control for different scenarios

## Comparison with Standard CAT

| Feature | Standard CAT | Enhanced CAT |
|---------|--------------|--------------|
| Optimization | Closed-form LS | Gradient-based |
| Loss Function | MSE | KL + Center + Separation |
| Centroid Updates | Static | EMA |
| Cluster Assignment | Fixed | Dynamic |
| Trust Region | No | Yes |
| Gradient Clipping | No | Yes |
| Convergence | Fast | Better quality |

## Tips for Usage

1. **Start with defaults**: Use default parameters for initial experiments
2. **Adjust learning rate**: Lower for stability, higher for faster convergence
3. **Monitor trust region**: Watch `lam` values to ensure PTQ preservation
4. **Cluster size**: Larger clusters may need more steps
5. **Reassignment frequency**: Higher frequency for dynamic datasets
6. **EMA decay**: Higher values for smoother updates

## References

- Original CAT paper and implementation
- Trust region methods in optimization
- EMA techniques in deep learning
- Dynamic clustering algorithms
