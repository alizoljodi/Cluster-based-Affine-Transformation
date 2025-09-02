#!/bin/bash

# Transformer Models Experiments Script
# This script runs experiments for ViT, Swin, and DeiT models

# Set your ImageNet data path here
DATA_PATH="/path/to/imagenet"

# Common parameters
BATCH_SIZE=32
WORKERS=4
NUM_SAMPLES=1024
SEED=42

echo "Starting Transformer Models Experiments"
echo "======================================"

# ViT Experiments
echo "Running ViT-Small experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch vit_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

echo "Running ViT-Base experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch vit_base \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

# Swin Experiments
echo "Running Swin-Small experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch swin_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

echo "Running Swin-Base experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch swin_base \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

# DeiT Experiments
echo "Running DeiT-Small experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch deit_small \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

echo "Running DeiT-Tiny experiments..."
python main_imagenet.py \
    --data_path $DATA_PATH \
    --arch deit_tiny \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --batch_size $BATCH_SIZE \
    --workers $WORKERS \
    --num_samples $NUM_SAMPLES \
    --seed $SEED \
    --alpha 0.4 \
    --num_clusters 64 \
    --pca_dim -1

echo "All transformer experiments completed!"
