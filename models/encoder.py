"""
encoder.py

Image Encoder for Mem3D.
Uses a ResNet-50 backbone.
"""

import torch.nn as nn
from torchvision import models

import config as cfg


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # Load a pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)

        # --- ResNet-50 Backbone ---
        # As per paper: "first three convolutional blocks of ResNet-50"
        # This corresponds to layers conv1, bn1, relu, maxpool, layer1, layer2, layer3
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # From paper: The output of layer3 in ResNet-50 is 1024 channels (not 512)
        # with a 28x28 feature map for a 224x224 input.
        # Let's check the paper's text again: "extract a 512x282 feature map"
        # This 512x28x28 is likely a typo in the paper.
        # Let's follow the ResNet-50 architecture:
        # Input: (B, 3, 224, 224)
        # layer1 out: (B, 256, 56, 56)
        # layer2 out: (B, 512, 28, 28)
        # layer3 out: (B, 1024, 14, 14)

        # Let's assume the paper meant "first *two* blocks" (layer1, layer2) to get 512 channels
        # Or, they used a custom ResNet.
        # Let's follow the paper's *text* (Sec 3.3) which is more specific than the diagram.
        # "extract a 512x28x28 feature map"
        # "The output channels of the three convolutional layers are 512, 256, and 256"

        # If we take ResNet-50's layer2 output, we get (B, 512, 28, 28). This matches.
        # So we will use ResNet-50 up to layer2.

        self.backbone_upto_layer2 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        # Output of backbone_upto_layer2 is (B, 512, 28, 28)

        # --- Custom Conv Layers (from Sec 3.3) ---
        # "followed by three sets of 2D convolutional layers, batch
        # normalization layers and ReLU layers."

        # Layer 1
        # "The kernel sizes... are 3x3, with a padding of 1."
        # "The output channels... are 512, 256, and 256"
        self.custom_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        # Output: (B, 512, 28, 28)

        # Layer 2
        # "There is a max pooling layer... after the second... ReLU layer"
        self.custom_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel 2x2
        )
        # Output: (B, 256, 14, 14)

        # Layer 3
        # "There is a max pooling layer... after the... third ReLU layer"
        self.custom_conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=cfg.FEATURE_DIM,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # cfg.FEATURE_DIM = 256
            nn.BatchNorm2d(cfg.FEATURE_DIM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel 2x2
        )
        # Output: (B, 256, 7, 7)

        # Final fully connected layer to flatten and create the feature vector
        self.fc = nn.Linear(cfg.FEATURE_DIM * 7 * 7, cfg.FEATURE_DIM)

    def forward(self, x):
        # x shape: (B, 3, 224, 224)

        # Pass through ResNet-50 backbone (up to layer2)
        x = self.backbone_upto_layer2(x)
        # x shape: (B, 512, 28, 28)

        # Pass through custom conv layers
        x = self.custom_conv1(x)
        # x shape: (B, 512, 28, 28)
        x = self.custom_conv2(x)
        # x shape: (B, 256, 14, 14)
        x = self.custom_conv3(x)
        # x shape: (B, 256, 7, 7)

        # Flatten
        x = x.view(x.size(0), -1)
        # x shape: (B, 256 * 7 * 7)

        # Final feature vector
        image_feature = self.fc(x)
        # image_feature shape: (B, 256)

        return image_feature
