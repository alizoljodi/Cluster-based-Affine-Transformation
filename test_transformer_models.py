#!/usr/bin/env python3
"""
Test script to verify that the new transformer models can be loaded and run correctly.
"""

import torch
import torch.nn as nn
import sys
import os

def test_model_loading():
    """Test that all transformer models can be loaded correctly."""
    print("Testing Transformer Models Loading")
    print("="*50)
    
    # Add current directory to path to import hubconf
    sys.path.append(os.getcwd())
    
    try:
        import hubconf
        print("✓ hubconf imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import hubconf: {e}")
        return False
    
    # Define models to test
    models_to_test = [
        'vit_small',
        'vit_base', 
        'swin_small',
        'swin_base',
        'deit_small',
        'deit_tiny'
    ]
    
    success_count = 0
    total_count = len(models_to_test)
    
    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        
        try:
            # Get model function
            model_func = getattr(hubconf, model_name)
            print(f"  ✓ Model function found")
            
            # Create model (without pretrained weights)
            model = model_func(pretrained=False)
            print(f"  ✓ Model created successfully")
            
            # Check model structure
            print(f"  - Model type: {type(model).__name__}")
            print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(2, 3, 224, 224)
            
            # Handle different output formats
            try:
                output = model(dummy_input)
                if isinstance(output, tuple):
                    # DeiT returns (cls_output, dist_output)
                    cls_output, dist_output = output
                    print(f"  ✓ Forward pass successful (DeiT - cls: {cls_output.shape}, dist: {dist_output.shape})")
                else:
                    print(f"  ✓ Forward pass successful (output: {output.shape})")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Forward pass failed: {e}")
                
        except Exception as e:
            print(f"  ✗ Model creation failed: {e}")
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Successfully tested: {success_count}/{total_count} models")
    
    if success_count == total_count:
        print("✓ All transformer models working correctly!")
        return True
    else:
        print("✗ Some models failed to load")
        return False

def test_model_architectures():
    """Test specific architectural features of the models."""
    print("\nTesting Model Architectures")
    print("="*50)
    
    try:
        import hubconf
        
        # Test ViT
        print("Testing ViT architecture...")
        vit = hubconf.vit_small(pretrained=False)
        
        # Check for key components
        assert hasattr(vit, 'patch_embed'), "ViT missing patch_embed"
        assert hasattr(vit, 'cls_token'), "ViT missing cls_token"
        assert hasattr(vit, 'pos_embed'), "ViT missing pos_embed"
        assert hasattr(vit, 'blocks'), "ViT missing transformer blocks"
        assert hasattr(vit, 'head'), "ViT missing classification head"
        print("✓ ViT architecture components verified")
        
        # Test Swin
        print("Testing Swin architecture...")
        swin = hubconf.swin_small(pretrained=False)
        
        # Check for key components
        assert hasattr(swin, 'patch_embed'), "Swin missing patch_embed"
        assert hasattr(swin, 'layers'), "Swin missing layers"
        assert hasattr(swin, 'norm'), "Swin missing norm"
        assert hasattr(swin, 'head'), "Swin missing classification head"
        print("✓ Swin architecture components verified")
        
        # Test DeiT
        print("Testing DeiT architecture...")
        deit = hubconf.deit_small(pretrained=False)
        
        # Check for key components
        assert hasattr(deit, 'patch_embed'), "DeiT missing patch_embed"
        assert hasattr(deit, 'cls_token'), "DeiT missing cls_token"
        assert hasattr(deit, 'dist_token'), "DeiT missing dist_token"
        assert hasattr(deit, 'blocks'), "DeiT missing transformer blocks"
        assert hasattr(deit, 'head'), "DeiT missing classification head"
        assert hasattr(deit, 'head_dist'), "DeiT missing distillation head"
        print("✓ DeiT architecture components verified")
        
        print("✓ All architecture tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Architecture test failed: {e}")
        return False

def test_quantization_compatibility():
    """Test that models are compatible with quantization framework."""
    print("\nTesting Quantization Compatibility")
    print("="*50)
    
    try:
        import hubconf
        from quant import QuantModel
        
        # Test with a small model
        model = hubconf.vit_small(pretrained=False)
        
        # Define quantization parameters
        wq_params = {'n_bits': 4, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': 4, 'channel_wise': False, 'scale_method': 'mse',
                     'leaf_param': True, 'prob': 0.5}
        
        # Create quantized model
        qmodel = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
        print("✓ Quantized model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = qmodel(dummy_input)
        print(f"✓ Quantized forward pass successful (output: {output.shape})")
        
        print("✓ Quantization compatibility verified!")
        return True
        
    except Exception as e:
        print(f"✗ Quantization compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("Transformer Models Test Suite")
    print("="*60)
    
    # Run all tests
    test1 = test_model_loading()
    test2 = test_model_architectures()
    test3 = test_quantization_compatibility()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Model Loading: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Architecture: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Quantization: {'✓ PASS' if test3 else '✗ FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n🎉 All tests passed! Transformer models are ready for experiments.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    print("="*60)
