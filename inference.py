"""
inference.py
Script for single image 3D reconstruction
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config as cfg
from models.mem3d import Mem3D


def load_model(checkpoint_path, device):
    """Load trained Mem3D model"""
    model = Mem3D().to(torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


def preprocess_image(image_path):
    """Preprocess input image for inference"""
    transform = cfg.INPUT_IMG_TRANSFORM

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model, image_path, device):
    """Generate 3D voxels from single image"""
    # Preprocess
    image_tensor = preprocess_image(image_path).to(torch.device(device))

    # Inference
    with torch.no_grad():
        final_shape, image_feature, retrieved_indices, retrieved_scores = model(
            image_tensor
        )

        raw_voxels = final_shape.squeeze(0).cpu().numpy()
        binary_voxels = (raw_voxels > cfg.DECODER_PROB_THRESHOLD).astype(np.uint8)

    return {
        "binary_voxels": binary_voxels,
        "raw_voxels": raw_voxels,
        "image_feature": image_feature.cpu().numpy(),
        "retrieved_indices": retrieved_indices.cpu().numpy(),
        "retrieved_scores": retrieved_scores.cpu().numpy(),
    }


def visualize_prediction(image_path, prediction, save_path=None):
    """Visualize input image and predicted voxels"""
    fig = plt.figure(figsize=(15, 5))

    # Input image
    ax1 = fig.add_subplot(131)
    image = Image.open(image_path)
    ax1.imshow(image)
    ax1.set_title("Input Image")
    ax1.axis("off")

    # Raw voxels
    ax2 = fig.add_subplot(132, projection="3d")
    raw_voxels = prediction["raw_voxels"]

    # FIX: Create proper coordinates for voxel visualization
    # The voxel grid should have dimensions (n+1, n+1, n+1)
    x, y, z = np.indices(
        (raw_voxels.shape[0] + 1, raw_voxels.shape[1] + 1, raw_voxels.shape[2] + 1)
    )

    # Use the raw voxels data directly (shape should be (n, n, n))
    ax2.voxels(x, y, z, raw_voxels > 0.1, alpha=0.3)
    ax2.set_title("Raw Voxels (threshold: 0.1)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Binary voxels
    ax3 = fig.add_subplot(133, projection="3d")
    binary_voxels = prediction["binary_voxels"]

    # FIX: Create proper coordinates for binary voxels too
    x, y, z = np.indices(
        (
            binary_voxels.shape[0] + 1,
            binary_voxels.shape[1] + 1,
            binary_voxels.shape[2] + 1,
        )
    )

    ax3.voxels(x, y, z, binary_voxels > 0, alpha=0.7)
    ax3.set_title(f"Binary Voxels (threshold: {cfg.DECODER_PROB_THRESHOLD})")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")

    plt.show()


def save_voxels(prediction, save_path):
    """Save voxel data to file"""
    np.save(save_path, prediction["binary_voxels"])
    print(f"Voxels saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Single Image 3D Reconstruction")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--threshold", type=float, default=0.3, help="Voxel threshold")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save_visualization", type=str, help="Path to save visualization"
    )
    parser.add_argument("--save_voxels", type=str, help="Path to save voxel data")

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint_path, args.device)

    # Run prediction
    print(f"Processing image: {args.image_path}")
    prediction = predict(model, args.image_path, args.device)

    # Print stats
    binary_voxels = prediction["binary_voxels"]
    filled_voxels = np.sum(binary_voxels)
    total_voxels = binary_voxels.size
    fill_ratio = filled_voxels / total_voxels

    print(f"Results:")
    print(f"  - Voxel grid: {binary_voxels.shape}")
    print(f"  - Filled voxels: {filled_voxels}/{total_voxels} ({fill_ratio:.2%})")
    print(f"  - Avg similarity: {np.mean(prediction['retrieved_scores']):.4f}")

    # Visualize
    visualize_prediction(args.image_path, prediction, args.save_visualization)

    # Save if requested
    if args.save_voxels:
        save_voxels(prediction, args.save_voxels)


if __name__ == "__main__":
    main()
