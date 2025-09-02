#!/usr/bin/env python3
"""
Detailed Transformer Models Experiments Script
This script runs comprehensive experiments for ViT, Swin, and DeiT models
across different bitwidths and configurations.
"""

import subprocess
import sys
import os
from typing import List, Dict, Any

def run_experiment(arch: str, w_bits: int, a_bits: int, data_path: str, 
                  batch_size: int = 32, workers: int = 4, num_samples: int = 1024,
                  seed: int = 42, alpha: float = 0.4, num_clusters: int = 64,
                  pca_dim: int = -1, use_pfq: bool = False) -> bool:
    """
    Run a single experiment with the given parameters.
    
    Args:
        arch: Model architecture
        w_bits: Weight quantization bits
        a_bits: Activation quantization bits
        data_path: Path to ImageNet dataset
        batch_size: Batch size for training
        workers: Number of workers for data loading
        num_samples: Number of calibration samples
        seed: Random seed
        alpha: Alpha blending value for CAT
        num_clusters: Number of clusters for CAT
        pca_dim: PCA dimension (-1 to disable)
        use_pfq: Whether to use PFQ
    
    Returns:
        True if experiment completed successfully, False otherwise
    """
    
    cmd = [
        "python", "main_imagenet.py",
        "--data_path", data_path,
        "--arch", arch,
        "--n_bits_w", str(w_bits),
        "--n_bits_a", str(a_bits),
        "--batch_size", str(batch_size),
        "--workers", str(workers),
        "--num_samples", str(num_samples),
        "--seed", str(seed),
        "--alpha", str(alpha),
        "--num_clusters", str(num_clusters),
        "--pca_dim", str(pca_dim)
    ]
    
    if use_pfq:
        cmd.extend([
            "--use_pfq",
            "--feature_steps", "1000",
            "--weight_steps", "4000",
            "--lr_feature", "4e-5",
            "--lr_weight", "1e-3",
            "--pfq_T", "1.0",
            "--pfq_lam", "1.0",
            "--qdrop_p", "0.1"
        ])
    
    print(f"Running experiment: {arch} W{w_bits}A{a_bits} {'(PFQ)' if use_pfq else ''}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Experiment completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_transformer_experiments(data_path: str, use_pfq: bool = False):
    """
    Run comprehensive experiments for transformer models.
    
    Args:
        data_path: Path to ImageNet dataset
        use_pfq: Whether to use PFQ for experiments
    """
    
    # Define model configurations
    models = [
        "vit_small", "vit_base",
        "swin_small", "swin_base", 
        "deit_small", "deit_tiny"
    ]
    
    # Define bitwidth configurations
    bit_configs = [
        (4, 4),  # W4A4
        (3, 3),  # W3A3
        (2, 2),  # W2A2
    ]
    
    # Define CAT configurations
    cat_configs = [
        {"alpha": 0.4, "num_clusters": 64, "pca_dim": -1},
        {"alpha": 0.6, "num_clusters": 128, "pca_dim": -1},
        {"alpha": 0.8, "num_clusters": 256, "pca_dim": -1},
    ]
    
    print("="*80)
    print("TRANSFORMER MODELS EXPERIMENTS")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"PFQ enabled: {use_pfq}")
    print(f"Models: {', '.join(models)}")
    print(f"Bit configs: {bit_configs}")
    print(f"CAT configs: {len(cat_configs)}")
    print("="*80)
    
    total_experiments = len(models) * len(bit_configs) * len(cat_configs)
    completed_experiments = 0
    failed_experiments = 0
    
    for model in models:
        print(f"\n{'='*20} {model.upper()} {'='*20}")
        
        for w_bits, a_bits in bit_configs:
            print(f"\n--- W{w_bits}A{a_bits} ---")
            
            for i, cat_config in enumerate(cat_configs):
                print(f"CAT config {i+1}/{len(cat_configs)}: {cat_config}")
                
                success = run_experiment(
                    arch=model,
                    w_bits=w_bits,
                    a_bits=a_bits,
                    data_path=data_path,
                    batch_size=32,
                    workers=4,
                    num_samples=1024,
                    seed=42,
                    alpha=cat_config["alpha"],
                    num_clusters=cat_config["num_clusters"],
                    pca_dim=cat_config["pca_dim"],
                    use_pfq=use_pfq
                )
                
                if success:
                    completed_experiments += 1
                else:
                    failed_experiments += 1
                
                print(f"Progress: {completed_experiments + failed_experiments}/{total_experiments}")
    
    print("\n" + "="*80)
    print("EXPERIMENTS SUMMARY")
    print("="*80)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed_experiments}")
    print(f"Failed: {failed_experiments}")
    print(f"Success rate: {completed_experiments/total_experiments*100:.1f}%")
    print("="*80)

def run_quick_test(data_path: str):
    """
    Run a quick test with one configuration for each model.
    
    Args:
        data_path: Path to ImageNet dataset
    """
    
    print("Running quick test with one configuration per model...")
    
    models = ["vit_small", "swin_small", "deit_small"]
    
    for model in models:
        print(f"\nTesting {model}...")
        success = run_experiment(
            arch=model,
            w_bits=4,
            a_bits=4,
            data_path=data_path,
            batch_size=16,  # Smaller batch for quick test
            workers=2,
            num_samples=512,  # Fewer samples for quick test
            seed=42,
            alpha=0.4,
            num_clusters=64,
            pca_dim=-1,
            use_pfq=False
        )
        
        if success:
            print(f"✓ {model} test passed!")
        else:
            print(f"✗ {model} test failed!")

if __name__ == "__main__":
    # Check if data path is provided
    if len(sys.argv) < 2:
        print("Usage: python run_transformer_detailed.py <data_path> [--pfq] [--quick]")
        print("  <data_path>: Path to ImageNet dataset")
        print("  --pfq: Enable PFQ experiments")
        print("  --quick: Run quick test instead of full experiments")
        sys.exit(1)
    
    data_path = sys.argv[1]
    use_pfq = "--pfq" in sys.argv
    quick_test = "--quick" in sys.argv
    
    # Check if data path exists
    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist!")
        sys.exit(1)
    
    if quick_test:
        run_quick_test(data_path)
    else:
        run_transformer_experiments(data_path, use_pfq)
