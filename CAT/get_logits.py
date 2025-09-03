"""
Utilities to extract logits from models.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class get_logits:
    """
    Extract logits from both quantized and full-precision models on a random
    subset of samples taken from the provided dataloader's dataset.

    The subset selection is reproducible when a seed is provided.
    Returns concatenated logits tensors.
    """

    def __init__(self, q_model: nn.Module, fp_model: nn.Module, dataloader: DataLoader, device: torch.device,
                 num_samples: Optional[int] = None, seed: Optional[int] = None) -> None:
        self.q_model = q_model
        self.fp_model = fp_model
        self.dataloader = dataloader
        self.device = device
        self.num_samples = num_samples
        self.seed = seed

    def __call__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q_model = self.q_model
        fp_model = self.fp_model
        base_loader = self.dataloader
        device = self.device
        num_samples = self.num_samples
        seed = self.seed

        # Build a subset loader if sampling is requested

        q_model.eval()
        fp_model.eval()

        all_q, all_fp = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(self.dataloader):
                images = images.to(device)
                q_logits = q_model(images)
                fp_logits = fp_model(images)
                all_q.append(q_logits.cpu())
                all_fp.append(fp_logits.cpu())
                if i % 50 == 0:
                    print(f"[get_logits] Processed batch {i}...")

        all_q = torch.cat(all_q, dim=0)  # [N, C]
        all_fp = torch.cat(all_fp, dim=0)  # [N, C]
        print(f"[get_logits] Extraction complete. Shapes -> all_q: {tuple(all_q.shape)}, all_fp: {tuple(all_fp.shape)}")

        return all_q, all_fp


