#!/usr/bin/env python3
"""
Test script to verify PFQ integration is working properly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from quant.pfq_pd_calibrator import PFQ_PD_Calibrator
from quant.pfq_wrappers import QuantizedConv2dWrapper, QuantizedLinearWrapper

def test_pfq_integration():
    """Test PFQ integration with a simple model."""
    print("Testing PFQ integration...")
    
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
    
    # Create original model
    original_model = SimpleModel()
    original_model.eval()
    
    # Create a copy for PFQ
    pfq_model = copy.deepcopy(original_model)
    pfq_model.eval()
    
    # Test input
    test_input = torch.randn(2, 3, 8, 8)
    
    # Get original output
    with torch.no_grad():
        original_output = original_model(test_input)
    
    print(f"Original model output shape: {original_output.shape}")
    
    # Apply PFQ
    device = torch.device("cpu")
    pfq_calibrator = PFQ_PD_Calibrator(
        model=pfq_model,
        w_bit=4,
        a_bit=4,
        device=device
    )
    
    # Prepare model with PFQ wrappers
    replaced_layers = pfq_calibrator.prepare()
    print(f"PFQ: Replaced {len(replaced_layers)} layers")
    
    # Check if layers were actually replaced
    for name, module in pfq_model.named_modules():
        if isinstance(module, (QuantizedConv2dWrapper, QuantizedLinearWrapper)):
            print(f"Found PFQ wrapper: {name} -> {type(module).__name__}")
    
    # Test forward pass with PFQ model
    with torch.no_grad():
        pfq_output = pfq_model(test_input)
    
    print(f"PFQ model output shape: {pfq_output.shape}")
    print(f"Outputs are close: {torch.allclose(original_output, pfq_output, atol=1e-3)}")
    
    # Test quantization is actually applied
    print("\nTesting quantization application:")
    
    # Get a sample layer
    sample_layer = None
    for name, module in pfq_model.named_modules():
        if isinstance(module, QuantizedConv2dWrapper):
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
        
        print(f"Quantization applied: {not torch.allclose(quant_output, fp_output, atol=1e-3)}")
        print(f"FP output range: [{fp_output.min():.4f}, {fp_output.max():.4f}]")
        print(f"Quant output range: [{quant_output.min():.4f}, {quant_output.max():.4f}]")
    
    print("PFQ integration test completed!")

if __name__ == "__main__":
    test_pfq_integration()
