from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pfq_wrappers import replace_with_pfq_wrappers, set_wrappers_mode

def pd_loss_from_logits(logits_q, logits_fp, T: float = 1.0, lam: float = 1.0):
    """Prediction Difference loss using KL divergence between logits."""
    return F.kl_div(
        F.log_softmax(logits_q / T, dim=1),
        F.softmax(logits_fp / T, dim=1),
        reduction="batchmean"
    ) / lam

class PFQ_PD_Calibrator:
    def __init__(self, model: nn.Module, w_bit: int, a_bit: int, device=None):
        self.model = model
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.replaced: Dict[str, nn.Module] = {}

    def prepare(self):
        """Prepare the model by replacing layers with PFQ wrappers and freezing BatchNorm."""
        self.model.eval()
        self.model, self.replaced = replace_with_pfq_wrappers(self.model, self.w_bit, self.a_bit)
        
        # Freeze BatchNorm layers
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
        
        print(f"PFQ: Replaced {len(self.replaced)} layers with PFQ wrappers")
        return self.replaced

    def flao_pd_layer(self, wrapper: nn.Module, calib_loader,
                      feature_steps=1000, weight_steps=4000,
                      lr_feature=4e-5, lr_weight=1e-3,
                      T=1.0, lam=1.0, qdrop_p=0.0):
        """
        Feature-Loss-Aware Optimization for a single layer.
        
        Stage A: Feature-first - optimize LSQ step sizes (current layer only), weights FP
        Stage B: Weight-last - freeze LSQ; optimize AdaRound (current layer only)
        """
        print(f"PFQ: Starting FLAO calibration for layer")
        
        # Stage A: feature-first (optimize LSQ of wrapper)
        print(f"PFQ: Stage A - Feature optimization ({feature_steps} steps)")
        for p in wrapper.parameters(): 
            p.requires_grad = False
        for p in wrapper.lsq_act.parameters(): 
            p.requires_grad = True
        opt_f = torch.optim.Adam(wrapper.lsq_act.parameters(), lr=lr_feature)

        it = iter(calib_loader)
        for t in range(feature_steps):
            try:
                images, *rest = next(it)
            except StopIteration:
                it = iter(calib_loader)
                images, *rest = next(it)
            images = images.to(self.device, non_blocking=True)

            # Teacher logits (FP): everyone FP
            with torch.no_grad():
                set_wrappers_mode(self.model, act=False, weight=False)
                logits_fp = self.model(images)

            # Student: current layer act-quant (maybe dropped), weights FP; others FP
            set_wrappers_mode(self.model, act=False, weight=False)
            wrapper.use_weight_quant = False
            wrapper.use_act_quant = not (qdrop_p > 0 and torch.rand(()) < qdrop_p)

            logits_q = self.model(images)
            loss = pd_loss_from_logits(logits_q, logits_fp, T=T, lam=lam)

            opt_f.zero_grad(set_to_none=True)
            loss.backward()
            opt_f.step()

            if t % 500 == 0:
                print(f"PFQ: Stage A step {t}/{feature_steps}, loss: {loss.item():.6f}")

        # Stage B: weight-last (optimize AdaRound of wrapper)
        print(f"PFQ: Stage B - Weight optimization ({weight_steps} steps)")
        for p in wrapper.parameters(): 
            p.requires_grad = False
        for p in wrapper.ada_w.parameters(): 
            p.requires_grad = True
        opt_w = torch.optim.Adam(wrapper.ada_w.parameters(), lr=lr_weight)

        it = iter(calib_loader)
        for t in range(weight_steps):
            try:
                images, *rest = next(it)
            except StopIteration:
                it = iter(calib_loader)
                images, *rest = next(it)
            images = images.to(self.device, non_blocking=True)

            # Teacher logits (FP): everyone FP
            with torch.no_grad():
                set_wrappers_mode(self.model, act=False, weight=False)
                logits_fp = self.model(images)

            # Student: current layer act ON, weight ON; others FP
            set_wrappers_mode(self.model, act=False, weight=False)
            wrapper.use_act_quant = True
            wrapper.use_weight_quant = True

            logits_q = self.model(images)
            loss = pd_loss_from_logits(logits_q, logits_fp, T=T, lam=lam)

            opt_w.zero_grad(set_to_none=True)
            loss.backward()
            opt_w.step()

            if t % 1000 == 0:
                print(f"PFQ: Stage B step {t}/{weight_steps}, loss: {loss.item():.6f}")

    def calibrate_all_layers(self, calib_loader, feature_steps=1000, weight_steps=4000,
                           lr_feature=4e-5, lr_weight=1e-3, T=1.0, lam=1.0, qdrop_p=0.0):
        """Calibrate all layers using FLAO schedule."""
        print(f"PFQ: Starting calibration of {len(self.replaced)} layers")
        
        # Iterate layers in deterministic order
        for i, (name, wrapper) in enumerate(self.replaced.items()):
            print(f"PFQ: Calibrating layer {i+1}/{len(self.replaced)}: {name}")
            self.flao_pd_layer(wrapper, calib_loader,
                              feature_steps=feature_steps,
                              weight_steps=weight_steps,
                              lr_feature=lr_feature, lr_weight=lr_weight,
                              T=T, lam=lam, qdrop_p=qdrop_p)
            print(f"PFQ: Completed calibration of {name}")

    @torch.no_grad()
    def finalize(self):
        """Finalize the model by copying quantized weights to original layers."""
        print("PFQ: Finalizing quantized model")
        for m in self.model.modules():
            if hasattr(m, "finalize") and callable(m.finalize):
                m.finalize()
        return self.model
