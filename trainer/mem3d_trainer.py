"""
Mem3D Trainer .
This class manages the entire training and validation process,
including the crucial Memory Writer logic.
"""

import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cfg
from models.mem3d import Mem3D
from utils.losses import ReconstructionLoss, VoxelTripletLoss


class Mem3DTrainer:
    def __init__(self, model: Mem3D, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # --- Optimizer (from paper Sec 4.2) ---
        # "Adam optimizer with a beta1 of 0.9 and a beta2 of 0.999"
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE,
            betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2),
        )

        # --- Learning Rate Scheduler (from paper Sec 4.2) ---
        # "initial learning rate is set to 0.001 and decayed by 2 after 150 epochs"
        # We use a custom lambda function for this specific decay schedule
        lr_lambda = (
            lambda epoch: 1.0 if epoch < cfg.LR_DECAY_EPOCH else cfg.LR_DECAY_FACTOR
        )
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # --- Loss Functions ---
        self.recon_loss_fn = ReconstructionLoss().to(cfg.DEVICE)
        self.triplet_loss_fn = VoxelTripletLoss().to(cfg.DEVICE)

        # --- Logging ---
        self.log_dir = cfg.LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = None  # Placeholder for TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not installed. Skipping logging.")

        # --- Memory Writer Threshold (delta) ---
        self.delta = cfg.DELTA

    def _calculate_sv_batch(self, V_batch, V_memory_batch):
        """Calculates Sv for batches, not the whole memory."""
        # V_batch: (B, res, res, res)
        # V_memory_batch: (B, res, res, res)
        # Flatten to (B, -1)
        v_batch_flat = V_batch.view(V_batch.shape[0], -1)
        v_mem_flat = V_memory_batch.view(V_memory_batch.shape[0], -1)

        mse = torch.mean((v_batch_flat - v_mem_flat) ** 2, dim=1)  # Shape (B)
        sv_scores = 1.0 - mse
        return sv_scores  # Shape (B)

    def _memory_writer(self, image_features, true_voxels):
        """
        Implements the Memory Writer strategy .
        This function updates the memory network (model.memory_module)
        during the training step.

        Args:
            image_features (Tensor): (B, feature_dim)
            true_voxels (Tensor): (B, res, res, res)
        """
        memory_keys = self.model.memory_module.memory_keys
        memory_values = self.model.memory_module.memory_values
        memory_ages = self.model.memory_module.memory_ages

        # Detach features/voxels from graph, we don't backprop through the writer
        F_batch = image_features.detach()
        V_batch = true_voxels.detach()

        # --- Step 1: Find best key similarity (Sk) for each item in batch ---
        # n1 = arg max_i Sk(F, Ki)
        # cos_sim shape: (B, M)
        cos_sim = F.cosine_similarity(
            F_batch.unsqueeze(1), memory_keys.unsqueeze(0), dim=2
        )
        best_sk_scores, n1_indices = torch.max(cos_sim, dim=1)  # Shape (B)

        # --- Step 2: Get the corresponding values (Vn1) and calculate Sv ---
        # (B, res, res, res)
        Vn1_batch = memory_values[n1_indices]
        # Sv(V, Vn1), shape (B)
        sv_scores_n1 = self._calculate_sv_batch(V_batch, Vn1_batch)

        # --- Step 3: Identify "Similar" vs "New" examples ---
        # Similar if Sv(V, Vn1) >= delta
        similar_mask = sv_scores_n1 >= self.delta
        new_mask = ~similar_mask

        # Get indices for each case
        similar_indices_batch = torch.where(similar_mask)[0]
        new_indices_batch = torch.where(new_mask)[0]

        # --- Strategy 1: Update Similar Examples (Eq. 4) ---
        if similar_indices_batch.shape[0] > 0:
            # Get the memory indices (n1) for these "similar" items
            n1_to_update = n1_indices[similar_indices_batch]

            # Get the corresponding image features (F)
            F_to_update = F_batch[similar_indices_batch]

            # Get the old keys (Kn1)
            Kn1_old = memory_keys[n1_to_update]

            # Calculate new key: Kn1 = (F + Kn1) / ||F + Kn1||
            Kn1_new = F_to_update + Kn1_old
            Kn1_new = F.normalize(Kn1_new, p=2, dim=1)

            # Update memory
            memory_keys[n1_to_update] = Kn1_new
            memory_ages[n1_to_update] = 0  # Reset age

            # Increment age for all others
            age_mask = torch.ones_like(memory_ages, dtype=torch.bool)
            age_mask[n1_to_update] = False
            memory_ages[age_mask] += 1

        # --- Strategy 2: Update New Examples (Eq. 5-8) ---
        if new_indices_batch.shape[0] > 0:
            num_new = new_indices_batch.shape[0]

            # Find the "oldest" slots in memory to replace (no)
            # no = arg max_i (Ai)
            _, no_indices = torch.topk(memory_ages, k=num_new)

            # Get the features (F) and voxels (V) to write
            F_to_write = F_batch[new_indices_batch]
            V_to_write = V_batch[new_indices_batch]

            # Overwrite memory
            memory_keys[no_indices] = F_to_write
            memory_values[no_indices] = V_to_write
            memory_ages[no_indices] = 0  # Reset age

            # Increment age for all others
            age_mask = torch.ones_like(memory_ages, dtype=torch.bool)
            age_mask[no_indices] = False
            memory_ages[age_mask] += 1

    def _train_epoch(self, epoch):
        self.model.train()  # Set model to training mode
        total_loss = 0.0
        total_recon_loss = 0.0
        total_triplet_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS} [Train]")

        for batch in pbar:
            # Dataset compatibility: expect keys 'imgs' and 'label' (utils.dataset.R2N2Dataset)
            imgs = batch["imgs"].to(cfg.DEVICE)
            # If multiple views are present (B, V, C, H, W), use the first view
            if imgs.dim() == 5:
                images = imgs[:, 0, ...]
            else:
                images = imgs

            true_voxels = batch["label"].to(cfg.DEVICE)

            # Ensure voxel tensors are float in [0,1] for BCE
            true_voxels = true_voxels.float()

            # --- Forward Pass ---
            final_shape, image_feature, _, _ = self.model(images)

            # --- Calculate Losses (L = L_triplet + lambda * L_recon) ---
            recon_loss = self.recon_loss_fn(final_shape, true_voxels)

            triplet_loss = self.triplet_loss_fn(
                image_feature,
                true_voxels,
                self.model.memory_module.memory_keys,
                self.model.memory_module.memory_values,
            )

            loss = triplet_loss + cfg.LAMBDA_RECON * recon_loss

            # --- Backward Pass & Optimization ---
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # --- Memory Writer Step ---
            # This happens *after* the backward pass, using the features
            # from the forward pass.
            self._memory_writer(image_feature, true_voxels)

            # --- Logging ---
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_triplet_loss += triplet_loss.item()

            pbar.set_postfix(
                Loss=f"{loss.item():.4f}",
                Recon=f"{recon_loss.item():.4f}",
                Triplet=f"{triplet_loss.item():.4f}",
            )

        avg_loss = total_loss / len(self.train_loader)
        avg_recon_loss = total_recon_loss / len(self.train_loader)
        avg_triplet_loss = total_triplet_loss / len(self.train_loader)

        if self.writer:
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            self.writer.add_scalar("Loss/train_recon", avg_recon_loss, epoch)
            self.writer.add_scalar("Loss/train_triplet", avg_triplet_loss, epoch)
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch + 1} Train Avg Loss: {avg_loss:.4f} "
            f"(Recon: {avg_recon_loss:.4f}, Triplet: {avg_triplet_loss:.4f})"
        )

    def _val_epoch(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        total_recon_loss = 0.0
        # We don't typically calculate triplet loss or update memory in val

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{cfg.EPOCHS} [Val]")

        with torch.no_grad():
            for batch in pbar:
                imgs = batch["imgs"].to(cfg.DEVICE)
                if imgs.dim() == 5:
                    images = imgs[:, 0, ...]
                else:
                    images = imgs

                true_voxels = batch["label"].to(cfg.DEVICE)

                true_voxels = true_voxels.float()

                # --- Forward Pass ---
                final_shape, _, _, _ = self.model(images)

                # --- Calculate Loss ---
                recon_loss = self.recon_loss_fn(final_shape, true_voxels)

                total_recon_loss += recon_loss.item()
                pbar.set_postfix(Recon=f"{recon_loss.item():.4f}")

        avg_recon_loss = total_recon_loss / len(self.val_loader)

        if self.writer:
            self.writer.add_scalar("Loss/val_recon", avg_recon_loss, epoch)

        print(f"Epoch {epoch + 1} Val Avg Recon Loss: {avg_recon_loss:.4f}")
        return avg_recon_loss

    def fit(self):
        print("--- Starting Training ---")
        best_val_loss = float("inf")

        for epoch in range(cfg.EPOCHS):
            self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            self.scheduler.step()  # Update learning rate

            # --- Save Checkpoint (Best Model) ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(self.log_dir, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best model to {save_path} (Val Loss: {val_loss:.4f})")

            # --- Save Checkpoint (Periodic) ---
            if (epoch + 1) % cfg.SAVE_PERIOD == 0:
                save_path = os.path.join(self.log_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved checkpoint to {save_path}")

        if self.writer:
            self.writer.close()
        print("--- Training Finished ---")
