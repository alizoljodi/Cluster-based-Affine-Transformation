"""
Core CAT (Cluster-based Affine Transformation) APIs.
"""

from typing import Dict, Optional, Tuple, List, Union, Any
from enum import Enum

import torch
import numpy as np
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    MiniBatchKMeans,
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    MINIBATCH_KMEANS = "minibatch_kmeans"
    GMM = "gmm"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    SPECTRAL = "spectral"


class CAT:
    """
    Cluster-based Affine Transformation utilities.
    """

    def _create_cluster_model(
        self,
        algorithm: Union[str, ClusteringAlgorithm],
        num_clusters: int,
        **kwargs
    ) -> Any:
        """
        Create a clustering model based on the specified algorithm.
        
        Args:
            algorithm: Clustering algorithm name or enum
            num_clusters: Number of clusters (may be ignored for some algorithms)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Clustering model instance
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        if algorithm == ClusteringAlgorithm.KMEANS:
            return KMeans(
                n_clusters=num_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 10),
                max_iter=kwargs.get('max_iter', 300)
            )
        elif algorithm == ClusteringAlgorithm.MINIBATCH_KMEANS:
            return MiniBatchKMeans(
                n_clusters=num_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 3),
                max_iter=kwargs.get('max_iter', 100),
                batch_size=kwargs.get('batch_size', 1024)
            )
        elif algorithm == ClusteringAlgorithm.GMM:
            return GaussianMixture(
                n_components=num_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 1),
                max_iter=kwargs.get('max_iter', 100),
                covariance_type=kwargs.get('covariance_type', 'full')
            )
        elif algorithm == ClusteringAlgorithm.DBSCAN:
            return DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5),
                metric=kwargs.get('metric', 'euclidean')
            )
        elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            return AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage=kwargs.get('linkage', 'ward'),
                affinity=kwargs.get('affinity', 'euclidean')
            )
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            return SpectralClustering(
                n_clusters=num_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 10),
                affinity=kwargs.get('affinity', 'rbf'),
                gamma=kwargs.get('gamma', 1.0)
            )
        else:
            raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    def _fit_predict_cluster(
        self,
        cluster_model: Any,
        features: np.ndarray,
        algorithm: Union[str, ClusteringAlgorithm]
    ) -> np.ndarray:
        """
        Fit the clustering model and predict cluster assignments.
        
        Args:
            cluster_model: Clustering model instance
            features: Feature array [N, D]
            algorithm: Clustering algorithm name or enum
            
        Returns:
            Cluster IDs array [N]
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        if algorithm == ClusteringAlgorithm.GMM:
            cluster_ids = cluster_model.fit_predict(features)
        elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            cluster_ids = cluster_model.fit_predict(features)
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            cluster_ids = cluster_model.fit_predict(features)
        elif algorithm == ClusteringAlgorithm.DBSCAN:
            cluster_ids = cluster_model.fit_predict(features)
        else:  # KMeans, MiniBatchKMeans
            cluster_ids = cluster_model.fit_predict(features)
        
        return cluster_ids

    def _predict_cluster(
        self,
        cluster_model: Any,
        features: np.ndarray,
        algorithm: Union[str, ClusteringAlgorithm],
        cluster_centers: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict cluster assignments for new data.
        
        Args:
            cluster_model: Fitted clustering model
            features: Feature array [N, D]
            algorithm: Clustering algorithm name or enum
            cluster_centers: Pre-computed cluster centers [K, D] (for algorithms without predict)
            
        Returns:
            Cluster IDs array [N]
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        if algorithm == ClusteringAlgorithm.GMM:
            cluster_ids = cluster_model.predict(features)
        elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            # AgglomerativeClustering doesn't have predict
            # Use nearest cluster center if available, otherwise refit (not ideal but works)
            if cluster_centers is not None:
                distances = pairwise_distances(features, cluster_centers)
                cluster_ids = np.argmin(distances, axis=1)
            else:
                # Fallback: refit (not ideal for inference but works)
                cluster_ids = cluster_model.fit_predict(features)
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            # SpectralClustering doesn't have predict
            if cluster_centers is not None:
                distances = pairwise_distances(features, cluster_centers)
                cluster_ids = np.argmin(distances, axis=1)
            else:
                # Fallback: refit (not ideal for inference but works)
                cluster_ids = cluster_model.fit_predict(features)
        elif algorithm == ClusteringAlgorithm.DBSCAN:
            # DBSCAN doesn't have predict, refit on new data
            cluster_ids = cluster_model.fit_predict(features)
        else:  # KMeans, MiniBatchKMeans
            cluster_ids = cluster_model.predict(features)
        
        return cluster_ids

    def _get_num_clusters(
        self,
        cluster_model: Any,
        cluster_ids: np.ndarray,
        algorithm: Union[str, ClusteringAlgorithm]
    ) -> int:
        """
        Get the number of clusters from the model or cluster IDs.
        
        Args:
            cluster_model: Clustering model
            cluster_ids: Cluster assignment array
            algorithm: Clustering algorithm name or enum
            
        Returns:
            Number of clusters
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        if algorithm == ClusteringAlgorithm.DBSCAN:
            # DBSCAN can have noise points (-1), count unique clusters excluding -1
            unique_clusters = np.unique(cluster_ids)
            unique_clusters = unique_clusters[unique_clusters >= 0]
            return len(unique_clusters)
        elif algorithm == ClusteringAlgorithm.GMM:
            return cluster_model.n_components
        elif algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            return cluster_model.n_clusters_
        elif algorithm == ClusteringAlgorithm.SPECTRAL:
            return cluster_model.n_clusters
        else:  # KMeans, MiniBatchKMeans
            return cluster_model.n_clusters

    def build_cluster_affine(
        self,
        all_q: torch.Tensor,
        all_fp: torch.Tensor,
        num_clusters: int = 64,
        pca_dim: Optional[int] = None,
        algorithm: Union[str, ClusteringAlgorithm] = ClusteringAlgorithm.KMEANS,
        **clustering_kwargs
    ) -> Tuple[Any, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Optional[PCA]]:
        """
        Build cluster affine correction model from pre-extracted logits.
        
        Args:
            all_q: Quantized logits [N, D]
            all_fp: Full-precision logits [N, D]
            num_clusters: Number of clusters (may be ignored for some algorithms like DBSCAN)
            pca_dim: PCA dimension (None to disable)
            algorithm: Clustering algorithm to use. Options:
                - 'kmeans' (default): KMeans clustering
                - 'minibatch_kmeans': MiniBatchKMeans for large datasets
                - 'gmm': Gaussian Mixture Model
                - 'dbscan': DBSCAN density-based clustering
                - 'agglomerative': Agglomerative hierarchical clustering
                - 'spectral': Spectral clustering
            **clustering_kwargs: Additional parameters for the clustering algorithm:
                - For DBSCAN: eps, min_samples, metric
                - For GMM: covariance_type, n_init, max_iter
                - For Agglomerative: linkage, affinity
                - For Spectral: affinity, gamma
                - For KMeans/MiniBatchKMeans: n_init, max_iter, batch_size (MiniBatch only)
        
        Returns:
            cluster_model: Fitted clustering model
            gamma_dict: Per-cluster gamma parameters
            beta_dict: Per-cluster beta parameters
            pca: PCA model (if used)
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        algo_name = algorithm.value
        print(f"[CAT] Building cluster-affine model: algorithm={algo_name}, num_clusters={num_clusters}, pca_dim={pca_dim}.")
        
        # Optional PCA for clustering only
        pca: Optional[PCA] = None
        if pca_dim is not None and pca_dim < all_q.shape[1]:
            print("[CAT] Applying PCA before clustering...")
            pca = PCA(n_components=pca_dim, random_state=42)
            q_features = pca.fit_transform(all_q.detach().cpu().numpy())
            print(f"[CAT] PCA complete. Reduced dim: {q_features.shape[1]}")
        else:
            q_features = all_q.detach().cpu().numpy()

        # Create and fit clustering model
        cluster_model = self._create_cluster_model(
            algorithm=algorithm,
            num_clusters=num_clusters,
            random_state=42,
            **clustering_kwargs
        )
        cluster_ids = self._fit_predict_cluster(cluster_model, q_features, algorithm)
        
        # Get actual number of clusters (important for DBSCAN which may have fewer)
        actual_num_clusters = self._get_num_clusters(cluster_model, cluster_ids, algorithm)
        
        unique, counts = np.unique(cluster_ids, return_counts=True)
        print(f"[CAT] {algo_name.upper()} complete. Found {actual_num_clusters} clusters. Cluster sizes:")
        for u, c in zip(unique, counts):
            if int(u) == -1:
                print(f"  - Noise points: {int(c)} samples")
            else:
                print(f"  - Cluster {int(u)}: {int(c)} samples")

        # Compute cluster centers for algorithms that need them (Agglomerative, Spectral)
        cluster_centers: Optional[np.ndarray] = None
        if algorithm in [ClusteringAlgorithm.AGGLOMERATIVE, ClusteringAlgorithm.SPECTRAL]:
            unique_clusters = np.unique(cluster_ids)
            unique_clusters = unique_clusters[unique_clusters >= 0]  # Exclude noise
            centers_list = []
            for cid in unique_clusters:
                mask = cluster_ids == cid
                center = q_features[mask].mean(axis=0)
                centers_list.append(center)
            if centers_list:
                cluster_centers = np.array(centers_list)
                print(f"[CAT] Computed {len(centers_list)} cluster centers for inference.")
        elif algorithm in [ClusteringAlgorithm.KMEANS, ClusteringAlgorithm.MINIBATCH_KMEANS]:
            cluster_centers = cluster_model.cluster_centers_
        elif algorithm == ClusteringAlgorithm.GMM:
            cluster_centers = cluster_model.means_

        # For each cluster: learn gamma, beta (per-class)
        gamma_dict: Dict[int, torch.Tensor] = {}
        beta_dict: Dict[int, torch.Tensor] = {}
        
        # Handle noise points for DBSCAN (assign them to a default cluster)
        has_noise = -1 in cluster_ids
        if has_noise:
            print("[CAT] DBSCAN detected noise points. Assigning default identity transformation.")
            # Use the largest cluster's parameters for noise points, or identity
            noise_cluster_id = -1
            gamma_dict[noise_cluster_id] = torch.ones(all_q.shape[1], device=all_q.device, dtype=all_q.dtype)
            beta_dict[noise_cluster_id] = torch.zeros(all_q.shape[1], device=all_q.device, dtype=all_q.dtype)

        # Process all clusters (including noise as -1 if present)
        unique_clusters = np.unique(cluster_ids)
        for cid in unique_clusters:
            if cid == -1:
                continue  # Already handled above
            
            # Numpy boolean mask for the current cluster id
            idxs_np = (cluster_ids == cid)
            # Convert to torch boolean tensor on the same device as inputs
            idxs_t = torch.from_numpy(idxs_np).to(device=all_q.device, dtype=torch.bool)

            if idxs_t.sum().item() == 0:
                # Empty cluster, default to identity
                gamma_dict[int(cid)] = torch.ones(all_q.shape[1], device=all_q.device, dtype=all_q.dtype)
                beta_dict[int(cid)] = torch.zeros(all_q.shape[1], device=all_q.device, dtype=all_q.dtype)
                continue

            q_c = all_q[idxs_t]  # [Nc, C]
            fp_c = all_fp[idxs_t]  # [Nc, C]

            # Closed-form least squares: fp ≈ gamma * q + beta
            mean_q = q_c.mean(dim=0)
            mean_fp = fp_c.mean(dim=0)

            # Compute variance, avoid div by zero
            var_q = q_c.var(dim=0, unbiased=False)
            var_q = torch.clamp(var_q, min=1e-8)

            gamma = ((q_c - mean_q) * (fp_c - mean_fp)).mean(dim=0) / var_q
            beta = mean_fp - gamma * mean_q

            gamma_dict[int(cid)] = gamma
            beta_dict[int(cid)] = beta

        # Attach cluster centers to model for algorithms that need them
        if cluster_centers is not None:
            cluster_model.cluster_centers_ = cluster_centers

        print("[CAT] Fitted per-cluster affine parameters (gamma, beta).")
        return cluster_model, gamma_dict, beta_dict, pca

    def apply_cluster_affine(
        self,
        q_logits: torch.Tensor,
        cluster_model: Any,
        gamma_dict: Dict[int, torch.Tensor],
        beta_dict: Dict[int, torch.Tensor],
        pca: Optional[PCA] = None,
        alpha: float = 0.4,
        algorithm: Union[str, ClusteringAlgorithm] = ClusteringAlgorithm.KMEANS,
    ) -> torch.Tensor:
        """
        Apply per-cluster affine correction with optional PCA and alpha blending.
        
        Args:
            q_logits: Quantized logits [N, D]
            cluster_model: Fitted clustering model
            gamma_dict: Per-cluster gamma parameters
            beta_dict: Per-cluster beta parameters
            pca: PCA model (if used during training)
            alpha: Blending factor (0.0 = no correction, 1.0 = full correction)
            algorithm: Clustering algorithm used (for proper prediction method)
        
        Returns:
            Corrected logits [N, D]
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        q_np = q_logits.detach().cpu().numpy()

        # Apply same PCA as used during LUT building
        if pca is not None:
            q_np = pca.transform(q_np)

        # Get cluster centers if available (for algorithms without predict method)
        cluster_centers = getattr(cluster_model, 'cluster_centers_', None)
        
        # Predict cluster assignments
        cluster_ids = self._predict_cluster(cluster_model, q_np, algorithm, cluster_centers=cluster_centers)

        corrected = []
        for i, q in enumerate(q_logits):
            cid = int(cluster_ids[i])
            
            # Handle noise points from DBSCAN or missing clusters
            if cid not in gamma_dict:
                # Fallback to identity transformation
                if -1 in gamma_dict:
                    cid = -1  # Use noise cluster if available
                else:
                    # Use the first available cluster as fallback
                    cid = list(gamma_dict.keys())[0]
            
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

    def evaluate_cluster_affine_with_alpha(
        self,
        q_model: torch.nn.Module,
        fp_model: torch.nn.Module,
        cluster_model: Any,
        gamma_dict: Dict[int, torch.Tensor],
        beta_dict: Dict[int, torch.Tensor],
        dataloader,
        device: torch.device,
        pca: Optional[PCA] = None,
        alpha: float = 0.4,
        plot: bool = False,
        algorithm: Union[str, ClusteringAlgorithm] = ClusteringAlgorithm.KMEANS,
    ) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Evaluate cluster affine correction with alpha blending and collect logits for plotting.
        
        Args:
            q_model: Quantized model
            fp_model: Full-precision model
            cluster_model: Fitted clustering model
            gamma_dict: Per-cluster gamma parameters
            beta_dict: Per-cluster beta parameters
            dataloader: Data loader for evaluation
            device: Device for computation
            pca: PCA model (if used during training)
            alpha: Blending factor
            plot: Whether to plot cluster samples
            algorithm: Clustering algorithm used (for proper prediction method)
        
        Returns:
            top1_acc: Top-1 accuracy
            top5_acc: Top-5 accuracy
            all_q_logits: All quantized logits
            all_fp_logits: All full-precision logits
            all_corrected_logits: All corrected logits
            all_cluster_ids: All cluster assignments
        """
        if isinstance(algorithm, str):
            algorithm = ClusteringAlgorithm(algorithm.lower())
        
        q_model.eval()
        fp_model.eval()
        total_top1, total_top5, total = 0, 0, 0
        print(f"[CAT] Evaluating with alpha={alpha}, plot={plot}, algorithm={algorithm.value}.")
        
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

                corrected_logits = self.apply_cluster_affine(
                    q_logits, cluster_model, gamma_dict, beta_dict, 
                    pca=pca, alpha=alpha, algorithm=algorithm
                )

                # Store logits for plotting
                all_q_logits.append(q_logits.cpu())
                all_fp_logits.append(fp_logits.cpu())
                all_corrected_logits.append(corrected_logits.cpu())
                
                # Get cluster IDs for this batch
                q_np = q_logits.cpu().numpy()
                if pca is not None:
                    q_np = pca.transform(q_np)
                cluster_centers = getattr(cluster_model, 'cluster_centers_', None)
                cluster_ids = self._predict_cluster(cluster_model, q_np, algorithm, cluster_centers=cluster_centers)
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
            # Get number of clusters for plotting
            num_clusters = self._get_num_clusters(cluster_model, all_cluster_ids, algorithm)
            visualizer.plot_cluster_samples(all_q_logits, all_fp_logits, all_corrected_logits, all_cluster_ids, num_clusters)
        
        return top1_acc, top5_acc, all_q_logits, all_fp_logits, all_corrected_logits, all_cluster_ids


