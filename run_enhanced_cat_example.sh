#!/bin/bash

# Example script to run main_imagenet.py with Enhanced CAT
# This demonstrates how to use the enhanced CAT with advanced optimization

echo "Running Enhanced CAT with ResNet18..."
python main_imagenet.py \
    --arch resnet18 \
    --batch_size 64 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005 \
    --use_enhanced_cat \
    --cat_steps 400 \
    --cat_lr 4e-4 \
    --cat_lam 0.3 \
    --cat_mu 0.3 \
    --cat_tau 1.0 \
    --cat_eps 0.01 \
    --cat_rho 1e-4 \
    --cat_beta_ema 0.9 \
    --cat_reassign_freq 10 \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

echo "Running Enhanced CAT with DeiT Tiny (requires timm)..."
python main_imagenet.py \
    --arch deit_tiny_patch16_224 \
    --batch_size 32 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005 \
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
    --num_clusters 32 \
    --pca_dim -1

echo "Running Enhanced CAT with multiple configurations..."
python main_imagenet.py \
    --arch resnet18 \
    --batch_size 64 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005 \
    --use_enhanced_cat \
    --cat_steps 200 \
    --cat_lr 3e-4 \
    --cat_lam 0.5 \
    --cat_mu 0.1 \
    --cat_tau 0.8 \
    --cat_eps 0.005 \
    --cat_rho 2e-4 \
    --cat_beta_ema 0.85 \
    --cat_reassign_freq 20 \
    --alpha 0.3 0.4 0.5 \
    --num_clusters 32 64 128 \
    --pca_dim -1 16 32
