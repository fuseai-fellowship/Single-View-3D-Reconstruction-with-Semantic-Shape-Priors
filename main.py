"""
main.py
"""

import os

import torch
from torch.utils.data import DataLoader

import config as cfg
from models.mem3d import Mem3D
from trainer.mem3d_trainer import Mem3DTrainer
from utils.dataset import R2N2Dataset


def main():
    # --- Setup ---
    # Set device
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for logging and model saving
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = R2N2Dataset(
        root=cfg.DATA_DIR,
        transform=cfg.INPUT_IMG_TRANSFORM,
        model_portion=cfg.TRAIN_SPLIT,
        max_views=cfg.MAX_VIEWS,
        batch_size=cfg.BATCH_SIZE,
    )
    val_dataset = R2N2Dataset(
        root=cfg.DATA_DIR,
        transform=cfg.INPUT_IMG_TRANSFORM,
        model_portion=cfg.VAL_SPLIT,
        max_views=cfg.MAX_VIEWS,
        batch_size=cfg.BATCH_SIZE,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # --- Model Initialization ---
    print("Initializing model...")
    model = Mem3D().to(device)

    # --- Trainer Initialization ---
    print("Initializing trainer...")
    trainer = Mem3DTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device
    )


if __name__ == "__main__":
    main()
