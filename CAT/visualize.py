"""
Visualization helpers for CAT experiments.
"""

from typing import Optional
import torch
import numpy as np


class VIS:
    """
    Visualization utilities for CAT experiments.
    """

    def plot_cluster_samples(
        self,
        all_q_logits: torch.Tensor,
        all_fp_logits: torch.Tensor,
        all_corrected_logits: torch.Tensor,
        all_cluster_ids: np.ndarray,
        num_clusters: int,
        samples_per_cluster: int = 5,
    ) -> None:
        """
        Plot randomly selected samples from each cluster for visualization.
        """
        import matplotlib.pyplot as plt
        import random
        
        # Set random seed for reproducible plotting
        random.seed(42)
        np.random.seed(42)
        
        fig, axes = plt.subplots(2, (num_clusters + 1) // 2, figsize=(15, 8))
        if num_clusters == 1:
            axes = [axes]
        elif len(axes.shape) == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for cid in range(num_clusters):
            if cid >= len(axes):
                break
                
            # Find samples in this cluster
            cluster_mask = (all_cluster_ids == cid)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                axes[cid].text(0.5, 0.5, f'Cluster {cid}\n(Empty)', 
                              ha='center', va='center', transform=axes[cid].transAxes)
                axes[cid].set_xticks([])
                axes[cid].set_yticks([])
                continue
            
            # Sample random indices from this cluster
            n_samples = min(samples_per_cluster, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, size=n_samples, replace=False)
            
            # Get logits for these samples
            q_samples = all_q_logits[sample_indices].numpy()
            fp_samples = all_fp_logits[sample_indices].numpy()
            corrected_samples = all_corrected_logits[sample_indices].numpy()
            
            # Plot first few dimensions for visualization
            dims_to_plot = min(10, q_samples.shape[1])
            x_pos = np.arange(dims_to_plot)
            
            # Plot mean values
            axes[cid].plot(x_pos, q_samples[:, :dims_to_plot].mean(axis=0), 'r-', label='Quantized', alpha=0.7)
            axes[cid].plot(x_pos, fp_samples[:, :dims_to_plot].mean(axis=0), 'b-', label='Full Precision', alpha=0.7)
            axes[cid].plot(x_pos, corrected_samples[:, :dims_to_plot].mean(axis=0), 'g--', label='Corrected', alpha=0.7)
            
            axes[cid].set_title(f'Cluster {cid} (n={len(cluster_indices)})')
            axes[cid].set_xlabel('Logit Dimension')
            axes[cid].set_ylabel('Mean Logit Value')
            axes[cid].legend(fontsize=8)
            axes[cid].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_clusters, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Cluster-wise Logit Comparison', y=1.02)
        plt.show()
        
        print(f"Plotted samples from {num_clusters} clusters with {samples_per_cluster} samples each (where available)")


