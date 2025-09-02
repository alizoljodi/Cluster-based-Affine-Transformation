import torch
import torch.nn as nn
import torch.nn.functional as F
from .pfq_quantizers import LSQActivation, AdaRoundWeight

class QuantizedLinearWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, w_bit: int, a_bit: int):
        super().__init__()
        self.linear = linear
        self.lsq_act = LSQActivation(a_bit, per_channel=False)
        self.ada_w = AdaRoundWeight(self.linear.weight.data, bit=w_bit, per_channel=False)
        # runtime toggles
        self.use_act_quant = True
        self.use_weight_quant = True

    def forward(self, x, use_act_quant=None, use_weight_quant=None):
        if use_act_quant is None: use_act_quant = self.use_act_quant
        if use_weight_quant is None: use_weight_quant = self.use_weight_quant
        
        # Ensure quantizers are in training mode for proper initialization
        if use_act_quant:
            self.lsq_act.train()
        if use_weight_quant:
            self.ada_w.train()
        
        x_q = self.lsq_act(x) if use_act_quant else x
        w = self.ada_w(self.linear.weight) if use_weight_quant else self.linear.weight
        return F.linear(x_q, w, self.linear.bias)

    @torch.no_grad()
    def finalize(self):
        self.linear.weight.copy_(self.ada_w(self.linear.weight))

class QuantizedConv2dWrapper(nn.Module):
    def __init__(self, conv: nn.Conv2d, w_bit: int, a_bit: int):
        super().__init__()
        self.conv = conv
        self.lsq_act = LSQActivation(a_bit, per_channel=False)
        self.ada_w = AdaRoundWeight(self.conv.weight.data, bit=w_bit, per_channel=True, ch_axis=0)
        self.use_act_quant = True
        self.use_weight_quant = True

    def forward(self, x, use_act_quant=None, use_weight_quant=None):
        if use_act_quant is None: use_act_quant = self.use_act_quant
        if use_weight_quant is None: use_weight_quant = self.use_weight_quant
        
        # Ensure quantizers are in training mode for proper initialization
        if use_act_quant:
            self.lsq_act.train()
        if use_weight_quant:
            self.ada_w.train()
        
        x_q = self.lsq_act(x) if use_act_quant else x
        w = self.ada_w(self.conv.weight) if use_weight_quant else self.conv.weight
        return F.conv2d(x_q, w, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding,
                        dilation=self.conv.dilation, groups=self.conv.groups)

    @torch.no_grad()
    def finalize(self):
        self.conv.weight.copy_(self.ada_w(self.conv.weight))

def replace_with_pfq_wrappers(model: nn.Module, w_bit: int, a_bit: int, skip_first_last: bool = True):
    """
    Replace Conv2d and Linear layers with PFQ wrappers.
    
    Args:
        model: The model to wrap
        w_bit: Weight quantization bitwidth
        a_bit: Activation quantization bitwidth
        skip_first_last: Whether to skip first and last layers (keep them 8-bit)
    """
    name_to_module = dict(model.named_modules())
    replaced = {}
    
    # Get all layer names to identify first and last
    layer_names = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            layer_names.append(name)
    
    for name, m in list(name_to_module.items()):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = name_to_module.get(parent_name, None) if parent_name else model

        if isinstance(m, nn.Conv2d):
            # Skip first/last layers if requested
            if skip_first_last and (name == layer_names[0] or name == layer_names[-1]):
                # Use 8-bit for first/last layers
                w = QuantizedConv2dWrapper(m, w_bit=8, a_bit=8)
            else:
                w = QuantizedConv2dWrapper(m, w_bit, a_bit)
            setattr(parent, child_name, w)
            replaced[name] = w
        elif isinstance(m, nn.Linear):
            # Skip first/last layers if requested
            if skip_first_last and (name == layer_names[0] or name == layer_names[-1]):
                # Use 8-bit for first/last layers
                w = QuantizedLinearWrapper(m, w_bit=8, a_bit=8)
            else:
                w = QuantizedLinearWrapper(m, w_bit, a_bit)
            setattr(parent, child_name, w)
            replaced[name] = w
    
    return model, replaced

def set_wrappers_mode(model: nn.Module, act: bool, weight: bool):
    """Set activation and weight quantization mode for all PFQ wrappers in the model."""
    for m in model.modules():
        if hasattr(m, "use_act_quant") and hasattr(m, "use_weight_quant"):
            m.use_act_quant = act
            m.use_weight_quant = weight
