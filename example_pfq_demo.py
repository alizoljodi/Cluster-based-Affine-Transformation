#!/usr/bin/env python3
"""
Demonstration script showing PFQ working correctly.
This script simulates the PFQ integration without requiring ImageNet data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from quant.pfq_pd_calibrator import PFQ_PD_Calibrator

def create_simple_model():
    """Create a simple model for demonstration."""
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
    
    return SimpleModel()

def simulate_calibration_data(num_samples=100):
    """Simulate calibration data."""
    data = torch.randn(num_samples, 3, 8, 8)
    targets = torch.randint(0, 10, (num_samples,))
    return data, targets

def demonstrate_pfq():
    """Demonstrate PFQ working correctly."""
    print("PFQ Demonstration")
    print("="*50)
    
    # Create model
    model = create_simple_model()
    model.eval()
    
    # Create test input
    test_input = torch.randn(2, 3, 8, 8)
    
    # Get original output
    with torch.no_grad():
        original_output = model(test_input)
    
    print(f"Original model output shape: {original_output.shape}")
    print(f"Original model output range: [{original_output.min():.4f}, {original_output.max():.4f}]")
    
    # Create calibration data
    calib_data, calib_targets = simulate_calibration_data(50)
    
    # Apply PFQ
    print("\nApplying PFQ...")
    pfq_model = copy.deepcopy(model)
    pfq_model.eval()
    
    device = torch.device("cpu")
    pfq_calibrator = PFQ_PD_Calibrator(
        model=pfq_model,
        w_bit=4,
        a_bit=4,
        device=device
    )
    
    # Prepare model with PFQ wrappers
    replaced_layers = pfq_calibrator.prepare()
    print(f"PFQ: Replaced {len(replaced_layers)} layers with PFQ wrappers")
    
    # Check if layers were replaced
    wrapper_count = 0
    for name, module in pfq_model.named_modules():
        if hasattr(module, 'use_act_quant') and hasattr(module, 'use_weight_quant'):
            wrapper_count += 1
            print(f"Found PFQ wrapper: {name}")
    
    print(f"Total PFQ wrappers found: {wrapper_count}")
    
    # Test forward pass with PFQ model
    with torch.no_grad():
        pfq_output = pfq_model(test_input)
    
    print(f"PFQ model output shape: {pfq_output.shape}")
    print(f"PFQ model output range: [{pfq_output.min():.4f}, {pfq_output.max():.4f}]")
    
    # Test quantization is actually applied
    print("\nTesting quantization application:")
    
    # Get a sample layer
    sample_layer = None
    for name, module in pfq_model.named_modules():
        if hasattr(module, 'use_act_quant') and hasattr(module, 'use_weight_quant'):
            sample_layer = module
            print(f"Testing layer: {name}")
            break
    
    if sample_layer:
        # Test with quantization enabled
        sample_layer.use_act_quant = True
        sample_layer.use_weight_quant = True
        quant_output = sample_layer(test_input)
        
        # Test with quantization disabled
        sample_layer.use_act_quant = False
        sample_layer.use_weight_quant = False
        fp_output = sample_layer(test_input)
        
        quantization_applied = not torch.allclose(quant_output, fp_output, atol=1e-3)
        print(f"Quantization applied: {quantization_applied}")
        
        if quantization_applied:
            print("✓ PFQ is working correctly!")
            print(f"FP output range: [{fp_output.min():.4f}, {fp_output.max():.4f}]")
            print(f"Quant output range: [{quant_output.min():.4f}, {quant_output.max():.4f}]")
        else:
            print("❌ PFQ is NOT working!")
    
    print("\n" + "="*50)
    print("PFQ Demonstration Completed!")
    print("="*50)

if __name__ == "__main__":
    demonstrate_pfq()
