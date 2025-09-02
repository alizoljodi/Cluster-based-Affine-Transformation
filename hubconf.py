from collections import OrderedDict
from models.resnet import resnet18 as _resnet18
from models.resnet import resnet50 as _resnet50
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from models.mnasnet import mnasnet as _mnasnet
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
from models.vit import vit_small as _vit_small, vit_base as _vit_base
from models.swin import swin_small as _swin_small, swin_base as _swin_base
from models.deit import deit_small as _deit_small, deit_tiny as _deit_tiny
import torch
dependencies = ['torch']
model_path = {
    'resnet18': '/home/tmp/resnet18_imagenet.pth.tar',
    'resnet50': '/home/tmp/resnet50_imagenet.pth.tar',
    'mbv2': '/home/tmp/mobilenetv2.pth.tar',
    'reg600m': '/home/tmp/regnet_600m.pth.tar',
    'reg3200m': '/home/tmp/regnet_3200m.pth.tar',
    'mnasnet': '/home/tmp/mnasnet.pth.tar',
}


def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['resnet18'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def resnet50(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet50(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['resnet50'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['mbv2'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model


def regnetx_600m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_600m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['reg600m'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def regnetx_3200m(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _regnetx_3200m(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['reg3200m'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def mnasnet(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mnasnet(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path['mnasnet'], map_location='cpu')
        model.load_state_dict(checkpoint)
    return model


def vit_small(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('vit_small_patch16_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _vit_small(**kwargs)
    else:
        # Use local implementation
        model = _vit_small(**kwargs)
    return model


def vit_base(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _vit_base(**kwargs)
    else:
        # Use local implementation
        model = _vit_base(**kwargs)
    return model


def swin_small(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _swin_small(**kwargs)
    else:
        # Use local implementation
        model = _swin_small(**kwargs)
    return model


def swin_base(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _swin_base(**kwargs)
    else:
        # Use local implementation
        model = _swin_base(**kwargs)
    return model


def deit_small(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('deit_small_patch16_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _deit_small(**kwargs)
    else:
        # Use local implementation
        model = _deit_small(**kwargs)
    return model


def deit_tiny(pretrained=False, **kwargs):
    if pretrained:
        try:
            # Try to load from timm
            import timm
            model = timm.create_model('deit_tiny_patch16_224', pretrained=True)
        except ImportError:
            print("Warning: timm not available, using local implementation")
            model = _deit_tiny(**kwargs)
    else:
        # Use local implementation
        model = _deit_tiny(**kwargs)
    return model
