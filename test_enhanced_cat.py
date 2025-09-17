#!/usr/bin/env python3
"""
Test script to verify Enhanced CAT implementation
"""

import torch
import numpy as np
from CAT.enhanced_cat import EnhancedCAT

def test_enhanced_cat():
    """Test Enhanced CAT functionality"""
    print("Testing Enhanced CAT implementation...")
    
    # Create synthetic data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    B, D = 1000, 1000  # batch size, feature dimension
    K = 32  # number of clusters
    
    # Generate synthetic logits
    torch.manual_seed(42)
    z_q = torch.randn(B, D, device=device) * 0.5  # quantized logits
    z_fp = z_q + torch.randn(B, D, device=device) * 0.1  # full-precision logits (slightly different)
    
    print(f"Generated synthetic data: z_q shape: {z_q.shape}, z_fp shape: {z_fp.shape}")
    
    # Test Enhanced CAT
    cat = EnhancedCAT()
    
    try:
        print("\n1. Testing Enhanced CAT build process...")
        cluster_model, gamma_dict, beta_dict, pca, centroids = cat.build_cluster_affine_enhanced(
            z_q, z_fp,
            num_clusters=K,
            num_steps=50,  # Reduced for testing
            lr=1e-3,
            lam=0.3,
            mu=0.3,
            tau=1.0,
            eps=0.01,
            rho=1e-4,
            beta_ema=0.9,
            reassign_freq=5,
            device=device
        )
        
        print(f"   ✓ Enhanced CAT build successful")
        print(f"   ✓ Cluster model: {type(cluster_model)}")
        print(f"   ✓ Gamma dict keys: {len(gamma_dict)}")
        print(f"   ✓ Beta dict keys: {len(beta_dict)}")
        print(f"   ✓ Centroids shape: {centroids.shape}")
        
        # Test individual loss functions
        print("\n2. Testing loss functions...")
        
        # Create test data
        test_z = torch.randn(10, D, device=device)
        test_y = torch.randint(0, K, (10,), device=device)
        test_c = torch.randn(K, D, device=device)
        
        # Test KL loss
        kl_loss = cat._kl_to_fp(test_z, test_z + 0.1, T=1.0)
        print(f"   ✓ KL loss: {kl_loss.item():.6f}")
        
        # Test center loss
        center_loss = cat._center_loss(test_z, test_y, test_c)
        print(f"   ✓ Center loss: {center_loss.item():.6f}")
        
        # Test separation loss
        sep_loss = cat._sep_loss(test_c, tau=1.0)
        print(f"   ✓ Separation loss: {sep_loss.item():.6f}")
        
        # Test trust region update
        print("\n3. Testing trust region update...")
        lam_new = cat._trust_region_update(kl_loss, kl_loss * 0.9, eps=0.01, lam=0.3)
        print(f"   ✓ Trust region update: {lam_new:.4f}")
        
        # Test EMA centroid update
        print("\n4. Testing EMA centroid update...")
        cat._ema_update_centroids(test_c, test_z, test_y, beta_ema=0.9)
        print(f"   ✓ EMA centroid update successful")
        
        print("\n" + "="*50)
        print("Enhanced CAT implementation test completed successfully!")
        
    except Exception as e:
        print(f"   ✗ Error in Enhanced CAT: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_cat()
