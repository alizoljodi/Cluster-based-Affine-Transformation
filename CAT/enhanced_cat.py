"""
Enhanced CAT (Cluster-based Affine Transformation) with advanced optimization.
"""

from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class EnhancedCAT:
    """
    Enhanced Cluster-based Affine Transformation with advanced optimization.
    Includes trust region updates, EMA centroids, and cluster reassignment.
    """

    def build_cluster_affine_enhanced(
        self,
        all_q: torch.Tensor,
        all_fp: torch.Tensor,
        num_clusters: int = 64,
        pca_dim: Optional[int] = None,
        num_steps: int = 400,
        lr: float = 4e-4,
        lam: float = 0.3,
        mu: float = 0.3,
        nu: float = 0.0,
        tau: float = 1.0,
        eps: float = 0.01,
        rho: float = 1e-4,
        beta_ema: float = 0.9,
        reassign_freq: int = 10,
        device: Optional[torch.device] = None,
    ) -> Tuple[KMeans, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[PCA], torch.Tensor]:
        """
        Build enhanced cluster affine correction model with advanced optimization.
        
        Args:
            all_q: Quantized logits [B, D]
            all_fp: Full-precision logits [B, D]
            num_clusters: Number of clusters
            pca_dim: PCA dimension (None to disable)
            num_steps: Number of optimization steps
            lr: Learning rate
            lam: Center loss weight
            mu: Separation loss weight
            nu: Additional loss weight (reserved)
            tau: Temperature for separation loss
            eps: Trust region epsilon
            rho: Regularization weight
            beta_ema: EMA decay for centroids
            reassign_freq: Cluster reassignment frequency
            device: Device for computation
            
        Returns:
            cluster_model: KMeans model
            gamma_dict: Per-cluster gamma parameters
            beta_dict: Per-cluster beta parameters
            pca: PCA model (if used)
            centroids: Learned centroids [K, D]
        """
        if device is None:
            device = all_q.device
            
        print(f"[EnhancedCAT] Building enhanced cluster-affine model: num_clusters={num_clusters}, pca_dim={pca_dim}")
        print(f"[EnhancedCAT] Optimization: steps={num_steps}, lr={lr}, lam={lam}, mu={mu}")
        
        # Move to device
        z_q = all_q.to(device)
        z_fp = all_fp.to(device)
        B, D = z_q.size()
        
        # Optional PCA for clustering only
        pca: Optional[PCA] = None
        if pca_dim is not None and pca_dim < D:
            print("[EnhancedCAT] Applying PCA before clustering...")
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(z_q.detach().cpu().numpy())
            print(f"[EnhancedCAT] PCA complete. Reduced dim: {q_features.shape[1]}")
        else:
            q_features = z_q.detach().cpu().numpy()

        # Initial clustering
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        y = cluster_model.fit_predict(q_features)
        y = torch.from_numpy(y).to(device)
        
        # Initialize centroids
        c = torch.zeros(num_clusters, D, device=device)
        for k in range(num_clusters):
            idx = (y == k).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                c[k] = z_q[idx].mean(dim=0)
            else:
                c[k] = torch.randn(D, device=device) * 0.1
                
        print(f"[EnhancedCAT] Initial clustering complete. Cluster sizes:")
        unique, counts = torch.unique(y, return_counts=True)
        for u, count in zip(unique, counts):
            print(f"  - Cluster {int(u)}: {int(count)} samples")

        # Initialize parameters
        K, D = c.size()
        gamma = torch.ones(K, D, device=device, requires_grad=True)
        beta = torch.zeros(K, D, device=device, requires_grad=True)
        
        # Optimizer
        opt = torch.optim.Adam([gamma, beta], lr=lr, weight_decay=0.0)
        
        # Baseline PTQ alignment (no CAT)
        with torch.no_grad():
            L_ptq_baseline = self._kl_to_fp(z_q, z_fp, T=1.0)
            print(f"[EnhancedCAT] Baseline KL loss: {L_ptq_baseline.item():.6f}")
        
        # Optimization loop
        for step in range(num_steps):
            # Apply affine transformation
            z_corr = self._apply_affine(z_q, y, gamma, beta)
            
            # Compute losses
            L_ptq = self._kl_to_fp(z_corr, z_fp, T=1.0)
            L_ctr = self._center_loss(z_corr, y, c)
            L_sep = self._sep_loss(c, tau=tau)
            reg = rho * ((gamma - 1).pow(2).mean() + beta.pow(2).mean())
            
            # Total loss
            L = L_ptq + lam * (L_ctr + mu * L_sep) + reg
            
            # Backward pass
            opt.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_([gamma, beta], max_norm=1.0)
            opt.step()
            
            # Trust-region control to avoid hurting PTQ
            lam = self._trust_region_update(L_ptq, L_ptq_baseline, eps=eps, lam=lam)
            
            # EMA centroids on corrected logits
            with torch.no_grad():
                self._ema_update_centroids(c, z_corr, y, beta_ema=beta_ema)
            
            # Optional: reassign clusters every T steps
            if (step + 1) % reassign_freq == 0:
                with torch.no_grad():
                    # k-means assignment on corrected logits
                    d2 = torch.cdist(z_corr, c)  # [B,K]
                    y_new = d2.argmin(dim=1)
                    
                    # Only update if there are significant changes
                    if (y_new != y).sum().item() > B * 0.05:  # 5% threshold
                        y = y_new
                        print(f"[EnhancedCAT] Step {step+1}: Reassigned clusters")
            
            # Logging
            if (step + 1) % 50 == 0:
                print(f"[EnhancedCAT] Step {step+1}/{num_steps}: "
                      f"L_ptq={L_ptq.item():.6f}, L_ctr={L_ctr.item():.6f}, "
                      f"L_sep={L_sep.item():.6f}, lam={lam:.4f}")
        
        # Convert to per-cluster dictionaries for compatibility
        gamma_dict: Dict[int, torch.Tensor] = {}
        beta_dict: Dict[int, torch.Tensor] = {}
        
        for k in range(num_clusters):
            gamma_dict[k] = gamma[k].detach()
            beta_dict[k] = beta[k].detach()
        
        print("[EnhancedCAT] Enhanced optimization complete.")
        return cluster_model, gamma_dict, beta_dict, pca, c.detach()

    def _apply_affine(self, z: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation: z * gamma[y] + beta[y]"""
        return z * gamma[y] + beta[y]

    def _kl_to_fp(self, z_corr: torch.Tensor, z_fp: torch.Tensor, T: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        """KL divergence loss between corrected and full-precision logits"""
        p = torch.softmax(z_fp / T, dim=-1).clamp_min(eps)
        q = torch.softmax(z_corr / T, dim=-1).clamp_min(eps)
        return (p * (p.log() - q.log())).sum(dim=-1).mean()

    def _center_loss(self, z_corr: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Center loss: minimize distance to cluster centroids"""
        return ((z_corr - c[y]).pow(2).sum(dim=1)).mean()

    def _sep_loss(self, c: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Separation loss: maximize distance between centroids"""
        diff = c[:, None, :] - c[None, :, :]
        dist = diff.pow(2).sum(dim=-1).sqrt()  # [K,K]
        mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
        return torch.exp(-dist[mask] / tau).mean()

    @torch.no_grad()
    def _ema_update_centroids(self, c: torch.Tensor, z_corr: torch.Tensor, y: torch.Tensor, beta_ema: float = 0.9):
        """EMA update of centroids"""
        for k in torch.unique(y):
            idx = (y == k).nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                c[k] = beta_ema * c[k] + (1 - beta_ema) * z_corr[idx].mean(dim=0)

    def _trust_region_update(self, L_ptq: torch.Tensor, L_ptq_baseline: torch.Tensor, 
                            eps: float, lam: float, decay: float = 0.5, min_lam: float = 1e-3) -> float:
        """Trust region update to avoid hurting PTQ performance"""
        if L_ptq.item() > L_ptq_baseline * (1.0 + eps):
            lam = max(min_lam, lam * decay)  # tighten trust region
        return lam

    def apply_cluster_affine_enhanced(
        self,
        q_logits: torch.Tensor,
        cluster_model: KMeans,
        gamma_dict: Dict[int, torch.Tensor],
        beta_dict: Dict[int, torch.Tensor],
        pca: Optional[PCA] = None,
        alpha: float = 0.4,
    ) -> torch.Tensor:
        """
        Apply enhanced per-cluster affine correction with optional PCA and alpha blending.
        """
        q_np = q_logits.detach().cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        cluster_ids = cluster_model.predict(q_np)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            gamma = gamma_dict[cid].to(q.device, dtype=q.dtype)
            beta = beta_dict[cid].to(q.device, dtype=q.dtype)
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

    def evaluate_cluster_affine_enhanced(
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
        Evaluate enhanced cluster affine correction with alpha blending.
        """
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        print(f"[EnhancedCAT] Evaluating with alpha={alpha}, plot={plot}.")
        
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

                corrected_logits = self.apply_cluster_affine_enhanced(
                    q_logits, cluster_model, gamma_dict, beta_dict, pca=pca, alpha=alpha
                )

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
        
        print(f"[EnhancedCAT Alpha={alpha:.2f}] Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"[EnhancedCAT Alpha={alpha:.2f}] Top-5 Accuracy: {top5_acc:.2f}%")
        
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
