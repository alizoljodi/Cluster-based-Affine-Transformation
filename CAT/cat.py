"""
Core CAT (Cluster-based Affine Transformation) APIs.
"""

from typing import Dict, Optional, Tuple, List

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class CAT:
    """
    Cluster-based Affine Transformation utilities.
    """

    def build_cluster_affine(
        self,
        all_q: torch.Tensor,
        all_fp: torch.Tensor,
        num_clusters: int = 64,
        pca_dim: Optional[int] = None,
    ) -> Tuple[KMeans, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[PCA]]:
        """
        Build cluster affine correction model from pre-extracted logits.
        """
        print(f"[CAT] Building cluster-affine model: num_clusters={num_clusters}, pca_dim={pca_dim}.")
        # Optional PCA for clustering only
        pca: Optional[PCA] = None
        if pca_dim is not None and pca_dim < all_q.shape[1]:
            print("[CAT] Applying PCA before clustering...")
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q.numpy())
            print(f"[CAT] PCA complete. Reduced dim: {q_features.shape[1]}")
        else:
            q_features = all_q.numpy()

        # Cluster quantized outputs
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_ids = cluster_model.fit_predict(q_features)
        unique, counts = np.unique(cluster_ids, return_counts=True)
        print("[CAT] KMeans complete. Cluster sizes:")
        for u, c in zip(unique, counts):
            print(f"  - Cluster {int(u)}: {int(c)} samples")

        # For each cluster: learn gamma, beta (per-class)
        gamma_dict: Dict[int, torch.Tensor] = {}
        beta_dict: Dict[int, torch.Tensor] = {}

        for cid in range(num_clusters):
            # Numpy boolean mask for the current cluster id
            idxs_np = (cluster_ids == cid)
            # Convert to torch boolean tensor on the same device as inputs
            idxs_t = torch.from_numpy(idxs_np).to(device=all_q.device, dtype=torch.bool)

            if idxs_t.sum().item() == 0:
                # Empty cluster, default to identity
                gamma_dict[cid] = torch.ones(all_q.shape[1])
                beta_dict[cid] = torch.zeros(all_q.shape[1])
                continue

            q_c = all_q[idxs_t]  # [Nc, C]
            fp_c = all_fp[idxs_t]  # [Nc, C]

            # Closed-form least squares: fp ≈ gamma * q + beta
            mean_q = q_c.mean(dim=0)
            mean_fp = fp_c.mean(dim=0)

            # Compute variance, avoid div by zero
            var_q = q_c.var(dim=0, unbiased=False)
            var_q[var_q < 1e-8] = 1e-8

            gamma = ((q_c - mean_q) * (fp_c - mean_fp)).mean(dim=0) / var_q
            beta = mean_fp - gamma * mean_q

            gamma_dict[cid] = gamma
            beta_dict[cid] = beta

        print("[CAT] Fitted per-cluster affine parameters (gamma, beta).")
        return cluster_model, gamma_dict, beta_dict, pca

    def apply_cluster_affine(
        self,
        q_logits: torch.Tensor,
        cluster_model: KMeans,
        gamma_dict: Dict[int, torch.Tensor],
        beta_dict: Dict[int, torch.Tensor],
        pca: Optional[PCA] = None,
        alpha: float = 0.4,
    ) -> torch.Tensor:
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        """
        q_np = q_logits.cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            gamma = gamma_dict[cid].to(q.device)
            beta = beta_dict[cid].to(q.device)
            affine_corrected = q * gamma + beta
            blended = q + alpha * (affine_corrected - q)
            corrected.append(blended)
        return torch.stack(corrected)

    def accuracy(self, output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def evaluate_cluster_affine_with_alpha(
        self,
        q_model: torch.nn.Module,
        fp_model: torch.nn.Module,
        cluster_model: KMeans,
        gamma_dict: Dict[int, torch.Tensor],
        beta_dict: Dict[int, torch.Tensor],
        dataloader,
        device: torch.device,
        pca: Optional[PCA] = None,
        alpha: float = 0.4,
        plot: bool = False,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Evaluate cluster affine correction with alpha blending and collect logits for plotting.
        """
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        print(f"[CAT] Evaluating with alpha={alpha}, plot={plot}.")
        
        # Store logits for plotting
        all_q_logits = []
        all_fp_logits = []
        all_corrected_logits = []
        all_cluster_ids = []

        with torch.no_grad():
            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                
                q_logits = q_model(images)
                fp_logits = fp_model(images)

                corrected_logits = self.apply_cluster_affine(q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha)

                # Store logits for plotting
                all_q_logits.append(q_logits.cpu())
                all_fp_logits.append(fp_logits.cpu())
                all_corrected_logits.append(corrected_logits.cpu())
                
                # Get cluster IDs for this batch
                q_np = q_logits.cpu().numpy()
                if pca is not None:
                    q_np = pca.transform(q_np)
                cluster_ids = cluster_model.predict(q_np)
                all_cluster_ids.append(cluster_ids)

                acc1, acc5 = self.accuracy(corrected_logits, targets, topk=(1, 5))
                total_top1 += acc1.item() * images.size(0)
                total_top5 += acc5.item() * images.size(0)
                total += images.size(0)

        top1_acc = total_top1 / total
        top5_acc = total_top5 / total
        
        print(f"[Alpha={alpha:.2f}] Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"[Alpha={alpha:.2f}] Top-5 Accuracy: {top5_acc:.2f}%")
        
        # Concatenate all logits and cluster IDs
        all_q_logits = torch.cat(all_q_logits, dim=0)
        all_fp_logits = torch.cat(all_fp_logits, dim=0)
        all_corrected_logits = torch.cat(all_corrected_logits, dim=0)
        all_cluster_ids = np.concatenate(all_cluster_ids, axis=0)
        
        # Plot randomly selected values from each cluster (optional)
        if plot:
            from .visualize import VIS
            visualizer = VIS()
            visualizer.plot_cluster_samples(all_q_logits, all_fp_logits, all_corrected_logits, all_cluster_ids, cluster_model.n_clusters)
        
        return top1_acc, top5_acc, all_q_logits, all_fp_logits, all_corrected_logits, all_cluster_ids


