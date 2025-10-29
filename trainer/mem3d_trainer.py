"""
mem3d_trainer.py

Mem3D Trainer
This class manages the entire training and validation process,
including the crucial Memory Writer logic.
"""

from torch.utils.data import DataLoader

import config as cfg
from models.mem3d import Mem3D


class Mem3DTrainer:
    def __init__(self, model: Mem3D, train_loader: DataLoader, val_loader: DataLoader):
        self.model = model.to(cfg.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
