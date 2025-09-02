#!/usr/bin/env python3
"""
Debug script to identify why PFQ is not applying quantization to layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.pfq_wrappers import replace_with_pfq_wrappers, set_wrappers_mode, QuantizedConv2dWrapper, QuantizedLinearWrapper
from quant.pfq_quantizers import LSQActivation, AdaRoundWeight

def debug_model_replacement():
    """Debug the model replacement process."""
    print("Debugging model replacement...")
    
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
    print(f"Original model: {model}")
    
    # Print all layers
    print("\nAll layers in model:")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            print(f"  {name}: {type(m).__name__}")
    
    # Try replacement without skip_first_last
    print("\nTrying replacement without skip_first_last...")
    try:
        model_copy = SimpleModel()
        model_copy, replaced = replace_with_pfq_wrappers(model_copy, w_bit=4, a_bit=4, skip_first_last=False)
        print(f"Success! Replaced {len(replaced)} layers")
        for name, wrapper in replaced.items():
            print(f"  {name}: {type(wrapper).__name__}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Try replacement with skip_first_last
    print("\nTrying replacement with skip_first_last...")
    try:
        model_copy2 = SimpleModel()
        model_copy2, replaced2 = replace_with_pfq_wrappers(model_copy2, w_bit=4, a_bit=4, skip_first_last=True)
        print(f"Success! Replaced {len(replaced2)} layers")
        for name, wrapper in replaced2.items():
            print(f"  {name}: {type(wrapper).__name__}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def debug_quantization():
    """Debug quantization application."""
    print("\nDebugging quantization application...")
    
    # Test LSQActivation
    x = torch.randn(2, 3, 4, 4)
    lsq = LSQActivation(bit=4, per_channel=False)
    lsq.train()
    output = lsq(x)
    print(f"LSQ input shape: {x.shape}, output shape: {output.shape}")
    print(f"LSQ quantization applied: {not torch.allclose(x, output)}")
    
    # Test AdaRoundWeight
    w = torch.randn(3, 4)
    ada_w = AdaRoundWeight(w, bit=4, per_channel=False)
    output_w = ada_w(w)
    print(f"AdaRound input shape: {w.shape}, output shape: {output_w.shape}")
    print(f"AdaRound quantization applied: {not torch.allclose(w, output_w)}")
    
    # Test wrapper
    conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    conv_wrapper = QuantizedConv2dWrapper(conv, w_bit=4, a_bit=4)
    x = torch.randn(2, 3, 8, 8)
    
    # Test with quantization enabled
    conv_wrapper.use_act_quant = True
    conv_wrapper.use_weight_quant = True
    output_q = conv_wrapper(x)
    output_fp = conv(x)
    print(f"Wrapper quantization applied: {not torch.allclose(output_q, output_fp)}")

if __name__ == "__main__":
    debug_model_replacement()
    debug_quantization()
