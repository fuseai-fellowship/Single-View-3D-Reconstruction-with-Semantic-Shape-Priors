"""
Main Mem3D Model.
This class assembles all the sub-modules:
1. Image Encoder
2. Memory Module
3. LSTM Shape Encoder
4. Shape Decoder
"""

import torch
import torch.nn as nn

import config as cfg
from models.encoder import Encoder
from models.memory_module import MemoryModule
from models.lstm_shape_encoder import LSTMShapeEncoder
from models.decoder import Decoder

class Mem3D(nn.Module):
    def __init__(self):
        super(Mem3D, self).__init__()
        
        self.image_encoder = Encoder()
        self.memory_module = MemoryModule()
        self.lstm_shape_encoder = LSTMShapeEncoder()
        self.decoder = Decoder()

    def forward(self, image):
        """
        Defines the complete forward pass for the Mem3D model.
        
        Args:
            image (Tensor): Batch of input images, shape (B, C, H, W)
        
        Returns:
            final_shape (Tensor): The final reconstructed 3D voxel grid.
                                  Shape: (B, res, res, res)
            image_feature (Tensor): The feature vector from the image encoder.
                                    Shape: (B, feature_dim)
            retrieved_indices (Tensor): Indices of the Top-K shapes from memory.
                                        Shape: (B, K)
            retrieved_scores (Tensor): Similarity scores of the Top-K shapes.
                                       Shape: (B, K)
        """
        
        # 1. Encode the input image to get a feature vector
        # Shape: (B, feature_dim)
        image_feature = self.image_encoder(image)
        
        # 2. Read from memory to get the Top-K similar 3D shapes
        # retrieved_shapes shape: (B, K, res, res, res)
        # retrieved_indices shape: (B, K)
        # retrieved_scores shape: (B, K)
        retrieved_shapes, retrieved_indices, retrieved_scores = self.memory_module.read(image_feature)
        
        # 3. Encode the sequence of 3D shapes with the LSTM
        # Shape: (B, lstm_hidden_dim)
        shape_prior_vector = self.lstm_shape_encoder(retrieved_shapes)
        
        # 4. Concatenate the image feature and the shape prior vector
        # This forms the final 1D vector to feed into the generative decoder
        # Shape: (B, feature_dim + lstm_hidden_dim)
        combined_vector = torch.cat([image_feature, shape_prior_vector], dim=1)
        
        # 5. Decode the combined vector to generate the final 3D shape
        # Shape: (B, res, res, res)
        final_shape = self.decoder(combined_vector)
        
        # Return all the necessary components for calculating losses
        return final_shape, image_feature, retrieved_indices, retrieved_scores

if __name__ == '__main__':
    # Test the full Mem3D model forward pass
    
    # Create a dummy batch of images
    dummy_images = torch.randn(cfg.BATCH_SIZE, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).to(cfg.DEVICE)
    
    model = Mem3D().to(cfg.DEVICE)
    model.eval() # Set to eval mode for testing
    
    print(f"Input image shape: {dummy_images.shape}")
    
    with torch.no_grad(): # No need to track gradients for a test pass
        final_shape, image_feature, indices, scores = model(dummy_images)
    
    print(f"Output final_shape shape: {final_shape.shape}")
    print(f"Output image_feature shape: {image_feature.shape}")
    print(f"Output retrieved_indices shape: {indices.shape}")
    print(f"Output retrieved_scores shape: {scores.shape}")
    
    assert final_shape.shape == (cfg.BATCH_SIZE, cfg.VOXEL_RES, cfg.VOXEL_RES, cfg.VOXEL_RES)
    assert image_feature.shape == (cfg.BATCH_SIZE, cfg.ENCODER_FEATURE_DIM)
    assert indices.shape == (cfg.BATCH_SIZE, cfg.TOP_K_MEMORY)
    assert scores.shape == (cfg.BATCH_SIZE, cfg.TOP_K_MEMORY)
    
    print("Mem3D full model test passed!")

