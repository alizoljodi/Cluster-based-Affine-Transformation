#!/bin/bash

# Example script to run main_imagenet.py with DeiT models
# This demonstrates how to use DeiT models instead of ResNet
# 
# Prerequisites: Install timm first
# pip install timm

echo "Running DeiT Tiny model..."
python main_imagenet.py \
    --arch deit_tiny_patch16_224 \
    --batch_size 32 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005

echo "Running DeiT Small model..."
python main_imagenet.py \
    --arch deit_small_patch16_224 \
    --batch_size 32 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005

echo "Running DeiT Base model..."
python main_imagenet.py \
    --arch deit_base_patch16_224 \
    --batch_size 16 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005

echo "Running DeiT Base Distilled model..."
python main_imagenet.py \
    --arch deit_base_distilled_patch16_224 \
    --batch_size 16 \
    --workers 4 \
    --data_path /datasets-to-imagenet \
    --n_bits_w 4 \
    --n_bits_a 4 \
    --num_samples 1024 \
    --seed 1005
