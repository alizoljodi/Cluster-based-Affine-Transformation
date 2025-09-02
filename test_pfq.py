#!/usr/bin/env python3
"""
Test script for PFQ (Prepositive Feature Quantization) integration.
This script tests the basic functionality of PFQ components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.pfq_quantizers import LSQActivation, AdaRoundWeight
from quant.pfq_wrappers import QuantizedLinearWrapper, QuantizedConv2dWrapper, replace_with_pfq_wrappers, set_wrappers_mode
from quant.pfq_pd_calibrator import PFQ_PD_Calibrator, pd_loss_from_logits
import copy

def test_lsq_activation():
    """Test LSQActivation quantizer."""
    print("Testing LSQActivation...")
    
    # Create test input
    x = torch.randn(2, 3, 4, 4)
    lsq = LSQActivation(bit=4, per_channel=False)
    
    # Test forward pass
    lsq.train()
    output = lsq(x)
    
    assert output.shape == x.shape
    assert not torch.allclose(output, x)  # Should be quantized
    print("✓ LSQActivation test passed")

def test_ada_round_weight():
    """Test AdaRoundWeight quantizer."""
    print("Testing AdaRoundWeight...")
    
    # Create test weight
    w = torch.randn(3, 4)
    ada_w = AdaRoundWeight(w, bit=4, per_channel=False)
    
    # Test forward pass
    output = ada_w(w)
    
    assert output.shape == w.shape
    # Check that at least some values are different (quantization occurred)
    diff_ratio = (output != w).float().mean().item()
    assert diff_ratio > 0.1, f"Quantization too weak, only {diff_ratio:.3f} values changed"
    print(f"✓ AdaRoundWeight test passed (quantization ratio: {diff_ratio:.3f})")

def test_wrappers():
    """Test PFQ wrappers."""
    print("Testing PFQ wrappers...")
    
    # Test Conv2d wrapper
    conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    conv_wrapper = QuantizedConv2dWrapper(conv, w_bit=4, a_bit=4)
    
    x = torch.randn(2, 3, 8, 8)
    output = conv_wrapper(x)
    
    assert output.shape == conv(x).shape
    print("✓ Conv2d wrapper test passed")
    
    # Test Linear wrapper
    linear = nn.Linear(10, 5)
    linear_wrapper = QuantizedLinearWrapper(linear, w_bit=4, a_bit=4)
    
    x = torch.randn(2, 10)
    output = linear_wrapper(x)
    
    assert output.shape == linear(x).shape
    print("✓ Linear wrapper test passed")

def test_pd_loss():
    """Test PD loss function."""
    print("Testing PD loss...")
    
    logits_q = torch.randn(4, 10)
    logits_fp = torch.randn(4, 10)
    
    loss = pd_loss_from_logits(logits_q, logits_fp, T=1.0, lam=1.0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    print("✓ PD loss test passed")

def test_model_replacement():
    """Test model replacement with PFQ wrappers."""
    print("Testing model replacement...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
            self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
            self.fc = nn.Linear(12 * 4 * 4, 10)
            
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.adaptive_avg_pool2d(x, (4, 4))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    original_model = copy.deepcopy(model)
    
    # Replace with PFQ wrappers
    model, replaced = replace_with_pfq_wrappers(model, w_bit=4, a_bit=4)
    
    assert len(replaced) == 3  # 2 conv + 1 linear
    print(f"✓ Model replacement test passed: {len(replaced)} layers replaced")

def test_calibrator():
    """Test PFQ calibrator initialization."""
    print("Testing PFQ calibrator...")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 6, 3, padding=1)
            self.fc = nn.Linear(6 * 4 * 4, 10)
            
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = F.adaptive_avg_pool2d(x, (4, 4))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    calibrator = PFQ_PD_Calibrator(model, w_bit=4, a_bit=4)
    
    # Test prepare method
    replaced = calibrator.prepare()
    assert len(replaced) == 2  # 1 conv + 1 linear
    print(f"✓ Calibrator test passed: {len(replaced)} layers prepared")

if __name__ == "__main__":
    print("Running PFQ integration tests...")
    print("="*50)
    
    try:
        test_lsq_activation()
        test_ada_round_weight()
        test_wrappers()
        test_pd_loss()
        test_model_replacement()
        test_calibrator()
        
        print("="*50)
        print("All tests passed! PFQ integration is working correctly.")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
