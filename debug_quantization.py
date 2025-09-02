#!/usr/bin/env python3
"""
Detailed debug script to understand why quantization is not being applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from quant.pfq_quantizers import LSQActivation, AdaRoundWeight
from quant.pfq_wrappers import QuantizedConv2dWrapper

def debug_lsq_activation():
    """Debug LSQActivation specifically."""
    print("Debugging LSQActivation...")
    
    x = torch.randn(2, 3, 4, 4)
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Input mean: {x.mean():.4f}")
    
    lsq = LSQActivation(bit=4, per_channel=False)
    print(f"LSQ initialized, s parameter: {lsq.s.data}")
    
    # Test in eval mode
    lsq.eval()
    output_eval = lsq(x)
    print(f"Eval mode - Input and output close: {torch.allclose(x, output_eval, atol=1e-3)}")
    print(f"Eval mode - Output range: [{output_eval.min():.4f}, {output_eval.max():.4f}]")
    
    # Test in train mode
    lsq.train()
    output_train = lsq(x)
    print(f"Train mode - Input and output close: {torch.allclose(x, output_train, atol=1e-3)}")
    print(f"Train mode - Output range: [{output_train.min():.4f}, {output_train.max():.4f}]")
    print(f"Train mode - s parameter after forward: {lsq.s.data}")
    
    # Check if quantization actually happened
    if not torch.allclose(x, output_train, atol=1e-3):
        print("✓ LSQ quantization is working!")
    else:
        print("❌ LSQ quantization is NOT working!")

def debug_ada_round_weight():
    """Debug AdaRoundWeight specifically."""
    print("\nDebugging AdaRoundWeight...")
    
    w = torch.randn(3, 4)
    print(f"Weight range: [{w.min():.4f}, {w.max():.4f}]")
    print(f"Weight mean: {w.mean():.4f}")
    
    ada_w = AdaRoundWeight(w, bit=4, per_channel=False)
    print(f"AdaRound initialized, sw parameter: {ada_w.sw.data}")
    
    output = ada_w(w)
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Input and output close: {torch.allclose(w, output, atol=1e-3)}")
    
    # Check if quantization actually happened
    if not torch.allclose(w, output, atol=1e-3):
        print("✓ AdaRound quantization is working!")
    else:
        print("❌ AdaRound quantization is NOT working!")

def debug_wrapper():
    """Debug the wrapper specifically."""
    print("\nDebugging QuantizedConv2dWrapper...")
    
    conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
    wrapper = QuantizedConv2dWrapper(conv, w_bit=4, a_bit=4)
    
    x = torch.randn(2, 3, 8, 8)
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    
    # Test with quantization disabled
    wrapper.use_act_quant = False
    wrapper.use_weight_quant = False
    output_fp = wrapper(x)
    print(f"FP output range: [{output_fp.min():.4f}, {output_fp.max():.4f}]")
    
    # Test with quantization enabled
    wrapper.use_act_quant = True
    wrapper.use_weight_quant = True
    output_quant = wrapper(x)
    print(f"Quant output range: [{output_quant.min():.4f}, {output_quant.max():.4f}]")
    
    # Check if quantization actually happened
    if not torch.allclose(output_fp, output_quant, atol=1e-3):
        print("✓ Wrapper quantization is working!")
    else:
        print("❌ Wrapper quantization is NOT working!")
        
        # Debug individual components
        print("\nDebugging individual components:")
        
        # Test LSQ separately
        wrapper.lsq_act.train()
        x_quant = wrapper.lsq_act(x)
        print(f"LSQ quantization applied: {not torch.allclose(x, x_quant, atol=1e-3)}")
        
        # Test AdaRound separately
        wrapper.ada_w.train()
        w_quant = wrapper.ada_w(conv.weight)
        print(f"AdaRound quantization applied: {not torch.allclose(conv.weight, w_quant, atol=1e-3)}")

if __name__ == "__main__":
    debug_lsq_activation()
    debug_ada_round_weight()
    debug_wrapper()
