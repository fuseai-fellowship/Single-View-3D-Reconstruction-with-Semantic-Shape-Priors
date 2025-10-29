"""
The Memory Module for Mem3D.
This class stores the (Key, Value, Age) triplets and implements
the "Memory Reader" logic.
The "Memory Writer" logic is implemented in the trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg

class MemoryModule(nn.Module):
    def __init__(self):
        super(MemoryModule, self).__init__()

        self.memory_size = cfg.MEMORY_SIZE
        self.feature_dim = cfg.ENCODER_FEATURE_DIM
        self.voxel_res = cfg.VOXEL_RES
        self.top_k = cfg.TOP_K_MEMORY

        # --- Initialize Memory Buffers ---
        # We use register_buffer so these tensors are part of the model's state
        # but are not updated by the optimizer (i.e., not model parameters).
        
        # Memory Keys (Image Features): (m, feature_dim)
        # We initialize with small random values.
        self.register_buffer(
            "memory_keys", 
            torch.randn(self.memory_size, self.feature_dim) * 0.01
        )
        
        # Memory Values (3D Voxel Grids): (m, res, res, res)
        # We initialize with zeros.
        self.register_buffer(
            "memory_values",
            torch.zeros(self.memory_size, self.voxel_res, self.voxel_res, self.voxel_res)
        )
        
        # Memory Ages: (m)
        # We initialize with large values so that new items can be written easily.
        self.register_buffer(
            "memory_ages",
            torch.full((self.memory_size,), float('inf'))
        )
        
        # Pointer to the next available empty slot (for initialization)
        self.register_buffer(
            "write_pointer",
            torch.tensor(0, dtype=torch.long)
        )

    def _calculate_key_similarity(self, query_features):
        """
        Calculates cosine similarity between query features and all memory keys.
        
        Args:
            query_features (Tensor): Batch of image features, shape (B, feature_dim)
        
        Returns:
            Tensor: Similarity matrix, shape (B, memory_size)
        """
        # Normalize both query and memory keys for cosine similarity
        query_features_norm = F.normalize(query_features, p=2, dim=1)
        memory_keys_norm = F.normalize(self.memory_keys, p=2, dim=1)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        # (B, feature_dim) @ (feature_dim, m) -> (B, m)
        similarity = torch.matmul(query_features_norm, memory_keys_norm.t())
        
        return similarity

    def read(self, image_features):
        """
        Implements the Memory Reader.
        Retrieves the Top-K most similar 3D shapes from memory.
        
        Args:
            image_features (Tensor): Batch of image features, shape (B, feature_dim)
        
        Returns:
            Tensor: Batch of retrieved 3D shapes, shape (B, K, res, res, res)
            Tensor: Indices of the retrieved shapes, shape (B, K)
            Tensor: Similarity scores of the retrieved shapes, shape (B, K)
        """
        batch_size = image_features.shape[0]

        # 1. Calculate similarity between image features and all memory keys
        # Shape: (B, m)
        similarities = self._calculate_key_similarity(image_features)

        # 2. Get the Top-K similarities and their indices
        # Shape: (B, K) for both
        top_k_similarities, top_k_indices = torch.topk(
            similarities, k=self.top_k, dim=1
        )
        
        # 3. Retrieve the corresponding 3D shapes (Values) from memory
        # We need to gather the values based on the indices.
        
        # Create a tensor to hold the output shapes
        # Shape: (B, K, res, res, res)
        retrieved_values = torch.zeros(
            batch_size, self.top_k, self.voxel_res, self.voxel_res, self.voxel_res,
            device=image_features.device
        )
        
        # Iterate over the batch to gather items
        # This is simpler than a complex batch-gather operation
        for i in range(batch_size):
            indices_for_batch_item = top_k_indices[i] # Shape: (K)
            retrieved_values[i] = self.memory_values[indices_for_batch_item]

        # The paper (Sec 3.2) states the value sequence is "ordered by the similarities"
        # torch.topk already returns them in descending order, so we are good.
        
        return retrieved_values, top_k_indices, top_k_similarities

    def forward(self, x):
        # The 'forward' method for this module is the 'read' operation
        return self.read(x)

if __name__ == '__main__':
    # Test the memory module
    memory = MemoryModule().to(cfg.DEVICE)
    
    # Create dummy image features
    test_features = torch.randn(cfg.BATCH_SIZE, cfg.ENCODER_FEATURE_DIM).to(cfg.DEVICE)
    
    # Test the read operation
    print("Testing memory read...")
    retrieved_shapes, indices, scores = memory.read(test_features)
    
    print(f"Input feature shape: {test_features.shape}")
    print(f"Retrieved shapes shape: {retrieved_shapes.shape}")
    print(f"Retrieved indices shape: {indices.shape}")
    print(f"Retrieved scores shape: {scores.shape}")
    
    assert retrieved_shapes.shape == (cfg.BATCH_SIZE, cfg.TOP_K_MEMORY, cfg.VOXEL_RES, cfg.VOXEL_RES, cfg.VOXEL_RES)
    assert indices.shape == (cfg.BATCH_SIZE, cfg.TOP_K_MEMORY)
    assert scores.shape == (cfg.BATCH_SIZE, cfg.TOP_K_MEMORY)
    
    print("Memory module test passed!")

