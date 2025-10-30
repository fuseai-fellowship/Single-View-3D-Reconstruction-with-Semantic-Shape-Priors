"""
LSTM Shape Encoder for Mem3D .
Takes a sequence of 3D voxel shapes and encodes them into
a single "shape prior vector".
"""

import torch
import torch.nn as nn

import config as cfg

class LSTMShapeEncoder(nn.Module):
    def __init__(self):
        super(LSTMShapeEncoder, self).__init__()
        
        self.input_dim = cfg.LSTM_INPUT_DIM   # 32*32*32 = 32768
        self.hidden_dim = cfg.LSTM_HIDDEN_DIM # 2048 (from Sec 3.3)
        self.num_layers = 1                   # 1 hidden layer (from Sec 3.3)
        self.top_k = cfg.TOP_K_MEMORY         # Length of the input sequence
        
        # LSTM layer
        # batch_first=True means input tensor shape is (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): A sequence of 3D shapes retrieved from memory.
                        Shape: (B, K, res, res, res)
        
        Returns:
            Tensor: The final "shape prior vector".
                    Shape: (B, hidden_dim)
        """
        batch_size = x.shape[0]
        
        # 1. Flatten the 3D voxel grids into 1D vectors
        # Input shape: (B, K, res, res, res)
        # We need to flatten the last 3 dims: (res, res, res) -> (res*res*res)
        x_flat = x.view(batch_size, self.top_k, -1)
        # x_flat shape: (B, K, 32768)
        
        if x_flat.shape[2] != self.input_dim:
            raise ValueError(f"Input voxel grid flattened dim {x_flat.shape[2]} does not match LSTM input_dim {self.input_dim}")

        # 2. Initialize hidden state and cell state for LSTM
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # 3. Pass the sequence through the LSTM
        # lstm_out shape: (B, K, hidden_dim) - output for each time step
        # hn shape: (num_layers, B, hidden_dim) - final hidden state
        # cn shape: (num_layers, B, hidden_dim) - final cell state
        lstm_out, (hn, cn) = self.lstm(x_flat, (h0, c0))
        
        # 4. Get the final hidden state
        # The paper (Sec 3.2) says it "outputs a fixed-length 'shape prior vector'".
        # This is typically the final hidden state of the LSTM.
        # Since num_layers is 1, hn shape is (1, B, hidden_dim).
        # We'll squeeze it to get (B, hidden_dim).
        shape_prior_vector = hn.squeeze(0)
        
        return shape_prior_vector

if __name__ == '__main__':
    # Test the LSTM Shape Encoder
    
    # Create a dummy sequence of 3D shapes
    # (Batch, K, res, res, res)
    dummy_seq = torch.randn(
        cfg.BATCH_SIZE, 
        cfg.TOP_K_MEMORY, 
        cfg.VOXEL_RES, 
        cfg.VOXEL_RES, 
        cfg.VOXEL_RES
    ).to(torch.device(cfg.DEVICE))
    
    lstm_encoder = LSTMShapeEncoder().to(torch.device(cfg.DEVICE))
    
    print(f"Input sequence shape: {dummy_seq.shape}")
    
    shape_prior = lstm_encoder(dummy_seq)
    
    print(f"Output shape prior vector shape: {shape_prior.shape}")
    
    assert shape_prior.shape == (cfg.BATCH_SIZE, cfg.LSTM_HIDDEN_DIM)
    
    print("LSTM Shape Encoder test passed!")

