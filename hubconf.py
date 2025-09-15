from collections import OrderedDict
from models.resnet import resnet18 as _resnet18
from models.resnet import resnet50 as _resnet50
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from models.mnasnet import mnasnet as _mnasnet
from models.regnet import regnetx_600m as _regnetx_600m
from models.regnet import regnetx_3200m as _regnetx_3200m
import torch
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. DeiT models will not work without timm installation.")
dependencies = ['torch']
model_path = {
    'resnet18': '/home/alz07xz/project/PD-Quant/pretrain/resnet18_imagenet.pth.tar',
    'resnet50': '/home/alz07xz/project/cluster/kmeans/resnet50/clustering/resnet50_imagenet.pth.tar',
    'mbv2': '/home/alz07xz/project/cluster/kmeans/resnet50/clustering/mobilenetv2.pth.tar',
    'reg600m': '/home/alz07xz/project/cluster/kmeans/resnet50/clustering/regnet_600m.pth.tar',
    'reg3200m': '/home/alz07xz/project/cluster/kmeans/resnet50/clustering/regnet_3200m.pth.tar',
    'mnasnet': '/home/alz07xz/project/cluster/kmeans/resnet50/clustering/mnasnet.pth.tar',
}
#


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


def deit_tiny_patch16_224(pretrained=False, **kwargs):
    # Use timm to create DeiT Tiny model
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for DeiT models. Please install it with: pip install timm")
    model = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained, **kwargs)
    return model


def deit_small_patch16_224(pretrained=False, **kwargs):
    # Use timm to create DeiT Small model
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for DeiT models. Please install it with: pip install timm")
    model = timm.create_model('deit_small_patch16_224', pretrained=pretrained, **kwargs)
    return model


def deit_base_patch16_224(pretrained=False, **kwargs):
    # Use timm to create DeiT Base model
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for DeiT models. Please install it with: pip install timm")
    model = timm.create_model('deit_base_patch16_224', pretrained=pretrained, **kwargs)
    return model


def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    # Use timm to create DeiT Base Distilled model
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required for DeiT models. Please install it with: pip install timm")
    model = timm.create_model('deit_base_distilled_patch16_224', pretrained=pretrained, **kwargs)
    return model
