"""
losses.py

Loss functions for Mem3D
1. ReconstructionLoss: Standard BCE for final 3D shape.
2. VoxelTripletLoss: Custom loss for training the encoder (Eq. 9-12).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class ReconstructionLoss(nn.Module):
    """
    Standard Binary Cross-Entropy loss for the final voxel grid.
    Compares the generated shape to the ground truth shape.
    This is L_recon in the paper.
    """

    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_voxels, true_voxels):
        """
        Args:
            pred_voxels (Tensor): Predicted voxels from Decoder,
                                  shape (B, res, res, res), values [0, 1]
            true_voxels (Tensor): Ground truth voxels from dataset,
                                  shape (B, res, res, res), values 0 or 1
        """
        # BCE loss expects (B, *) shape
        return self.bce_loss(pred_voxels, true_voxels)


class VoxelTripletLoss(nn.Module):
    """
    Voxel Triplet Loss (L_triplet) as described in Section 4.2.
    This loss trains the Image Encoder.
    It tries to minimize the distance between an anchor (image) and a positive
    (correct shape) and maximize the distance to a negative (incorrect shape).

    L_triplet = max(0, D(F, K_p) - D(F, K_n) + alpha)

    - Anchor (F): The image feature vector from the Encoder.
    - Positive (K_p): The key (feature) of a "positive" 3D shape from memory.
    - Negative (K_n): The key (feature) of a "negative" 3D shape from memory.
    - alpha: The margin (cfg.MARGIN).
    - D(a, b): Cosine similarity distance: 1 - cos_sim(a, b)

    Positive/Negative selection (Eq. 9, 10):
    - A memory item 'i' is POSITIVE if Sv(V, Vi) > beta
    - A memory item 'i' is NEGATIVE if Sv(V, Vi) < beta
    """

    def __init__(self):
        super(VoxelTripletLoss, self).__init__()
        self.margin = cfg.MARGIN  # alpha
        self.beta = cfg.BETA  # Sv threshold

        # We need a similarity metric for 3D shapes (Sv)
        # Using the one from the paper: Sv(V, Vi) = 1 - MSE(V, Vi)
        self.mse_loss = nn.MSELoss(reduction="mean")

    def _calculate_sv(self, V_batch, V_memory):
        """
        Calculates the 3D shape similarity (Sv) between a batch of
        ground truth shapes and all shapes in memory.

        Args:
            V_batch (Tensor): Ground truth 3D shapes, (B, res, res, res)
            V_memory (Tensor): All 3D shapes in memory, (M, res, res, res)

        Returns:
            Tensor: Pairwise Sv scores, shape (B, M)
        """
        B = V_batch.shape[0]
        M = V_memory.shape[0]

        # Reshape for pairwise MSE calculation
        # (B, 1, V_flat)
        v_batch_flat = V_batch.view(B, 1, -1)
        # (1, M, V_flat)
        v_mem_flat = V_memory.view(1, M, -1)

        # Calculate pairwise MSE: (v_batch - v_mem)^2 -> mean over last dim
        mse = torch.mean((v_batch_flat - v_mem_flat) ** 2, dim=2)  # Shape (B, M)

        # Sv = 1 - MSE
        sv_scores = 1.0 - mse
        return sv_scores  # Shape (B, M)

    def _cosine_distance(self, f1, f2):
        """Calculates pairwise cosine distance: 1 - cos_sim(f1, f2)"""
        cos_sim = F.cosine_similarity(f1.unsqueeze(1), f2.unsqueeze(0), dim=2)
        return 1.0 - cos_sim  # Shape (B, M)

    def forward(self, anchor_features, true_voxels, memory_keys, memory_values):
        """
        Args:
            anchor_features (Tensor): Image features from Encoder, (B, feature_dim)
            true_voxels (Tensor): Ground truth 3D shapes, (B, res, res, res)
            memory_keys (Tensor): All keys in memory, (M, feature_dim)
            memory_values (Tensor): All values in memory, (M, res, res, res)
        """
        # 1. Calculate all pairwise 3D shape similarities (Sv)
        # sv_scores shape: (B, M)
        sv_scores = self._calculate_sv(true_voxels, memory_values)

        # 2. Create masks for positive and negative examples
        # Positive mask: 1 where Sv > beta, 0 otherwise
        pos_mask = (sv_scores > self.beta).float()
        # Negative mask: 1 where Sv < beta, 0 otherwise
        neg_mask = (sv_scores < self.beta).float()

        # 3. Calculate all pairwise key similarities (Sk)
        # We need the distances D(F, K) = 1 - Sk(F, K)
        # key_dists shape: (B, M)
        key_dists = self._cosine_distance(anchor_features, memory_keys)

        # 4. Find the hardest positive and hardest negative for each anchor

        # --- Hardest Positive ---
        # We want the *positive* item with the *largest* distance (hardest)
        # Set non-positive distances to a very small number so they're not chosen
        pos_dists = key_dists.clone()
        pos_dists[pos_mask == 0] = -1e9
        hardest_pos_dist, _ = torch.max(pos_dists, dim=1)  # Shape (B)

        # --- Hardest Negative ---
        # We want the *negative* item with the *smallest* distance (hardest)
        # Set non-negative distances to a very large number
        neg_dists = key_dists.clone()
        neg_dists[neg_mask == 0] = 1e9
        hardest_neg_dist, _ = torch.min(neg_dists, dim=1)  # Shape (B)

        # 5. Calculate the Triplet Loss
        # L = max(0, D(F, K_p) - D(F, K_n) + margin)
        loss = torch.clamp(hardest_pos_dist - hardest_neg_dist + self.margin, min=0.0)

        # Handle cases with no positives/negatives
        # If a row had no positives, pos_mask.sum(1) == 0.
        # If a row had no negatives, neg_mask.sum(1) == 0.
        valid_triplets = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=anchor_features.device, requires_grad=True)

        # Only average the loss over valid triplets
        final_loss = loss[valid_triplets].mean()

        return final_loss
