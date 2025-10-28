"""
decoder.py

Shape Decoder for Mem3D.
This is a GENERATIVE network. It takes a 1D feature vector
and generates a 3D voxel grid using 3D Transposed Convolutions.
"""

import torch.nn as nn

import config as cfg


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.input_dim = cfg.DECODER_INPUT_DIM  # feature_dim + lstm_hidden_dim

        # We first need to reshape the 1D input vector into a small 3D feature map.
        # A common starting point for a 32x32x32 output is to start at 2x2x2.
        # So, we need a Linear layer to project `input_dim` to `C * 2 * 2 * 2`
        # Let's use the first channel number from the paper: 512
        self.start_channels = 512
        self.fc = nn.Linear(self.input_dim, self.start_channels * 2 * 2 * 2)

        # Now we define the 3D Transposed Convolution layers based on Sec 3.3
        # "five 3D transposed convolutional layers"
        # "output channel numbers... are 512, 128, 32, 8, and 1"
        # "first four... kernel sizes 4x4x4, with strides of 2 and paddings of 1"

        # Layer 1: (B, 512, 2, 2, 2) -> (B, 512, 4, 4, 4)
        # Paper says output channels are 512, but our input is already 512.
        # Let's assume the 512 is the *output* of the first layer.
        self.convT1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.start_channels,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

        # Layer 2: (B, 512, 4, 4, 4) -> (B, 128, 8, 8, 8)
        self.convT2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Layer 3: (B, 128, 8, 8, 8) -> (B, 32, 16, 16, 16)
        self.convT3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Layer 4: (B, 32, 16, 16, 16) -> (B, 8, 32, 32, 32)
        self.convT4 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        # Layer 5: (B, 8, 32, 32, 32) -> (B, 1, 32, 32, 32)
        # "next... has a bank of 1x1x1 filter"
        # "last... followed by a sigmoid function"
        self.convT5 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): The combined feature vector.
                        Shape: (B, DECODER_INPUT_DIM)

        Returns:
            Tensor: The final generated 3D voxel grid.
                    Shape: (B, 1, res, res, res)
        """
        batch_size = x.shape[0]

        # 1. Project and reshape 1D vector to 3D feature map
        x = self.fc(x)
        x = x.view(batch_size, self.start_channels, 2, 2, 2)
        # x shape: (B, 512, 2, 2, 2)

        # 2. Pass through 3D Transposed Conv layers
        x = self.convT1(x)  # (B, 512, 4, 4, 4)
        x = self.convT2(x)  # (B, 128, 8, 8, 8)
        x = self.convT3(x)  # (B, 32, 16, 16, 16)
        x = self.convT4(x)  # (B, 8, 32, 32, 32)
        x = self.convT5(x)  # (B, 1, 32, 32, 32)

        # 3. Squeeze the channel dimension for the loss function
        # Shape: (B, res, res, res)
        final_shape = x.squeeze(1)

        return final_shape
