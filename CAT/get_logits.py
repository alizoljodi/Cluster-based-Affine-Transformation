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
        if num_samples is not None and hasattr(base_loader, 'dataset'):
            dataset = base_loader.dataset
            dataset_len = len(dataset)
            k = min(int(num_samples), dataset_len)
            print(f"[get_logits] Sampling {k}/{dataset_len} examples from dataset (seed={seed}).")
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(seed))
            else:
                generator = None
            # Sample k unique indices reproducibly (if seed provided)
            if generator is None:
                indices = torch.randperm(dataset_len)[:k].tolist()
            else:
                indices = torch.randperm(dataset_len, generator=generator)[:k].tolist()
            subset = Subset(dataset, indices)
            dataloader = DataLoader(
                subset,
                batch_size=base_loader.batch_size,
                shuffle=False,
                num_workers=base_loader.num_workers,
                pin_memory=getattr(base_loader, 'pin_memory', False),
                drop_last=False,
                collate_fn=base_loader.collate_fn,
            )
            try:
                dl_len = len(dataloader)
            except TypeError:
                dl_len = 'unknown'
            print(f"[get_logits] Using subset DataLoader: batches={dl_len}, batch_size={base_loader.batch_size}.")
        else:
            dataloader = base_loader
            try:
                dl_len = len(dataloader)
            except TypeError:
                dl_len = 'unknown'
            print(f"[get_logits] Using full DataLoader: batches={dl_len}, batch_size={base_loader.batch_size}.")

        q_model.eval()
        fp_model.eval()

        all_q, all_fp = [], []

        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
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


