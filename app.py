import streamlit as st
import torch
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os

import config as cfg
from models.mem3d import Mem3D

# ------------- Helper functions -------------

def get_available_checkpoints(directory="checkpoints"):
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".pth")]

def load_model(checkpoint_path, device):
    model = Mem3D().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(img: Image.Image):
    transform = cfg.INPUT_IMG_TRANSFORM
    img = img.convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)

def predict(model, image_tensor, device):
    with torch.no_grad():
        final_shape, _, _, _ = model(image_tensor.to(device))
        raw_voxels = final_shape.squeeze(0).cpu().numpy()
        binary_voxels = (raw_voxels > cfg.DECODER_PROB_THRESHOLD)
    return raw_voxels, binary_voxels

def plot_voxels(voxel_grid, threshold=0.5, opacity=0.6):
    x, y, z = np.where(voxel_grid > threshold)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=3, color=z, colorscale="Viridis", opacity=opacity)
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig

# ------------- Streamlit App -------------

st.title("Mem3D: Single-View 3D Reconstruction")
st.write(
    "Select a trained model checkpoint from 'checkpoints/', "
    "upload a single image, and reconstruct its 3D voxel grid!"
)

# Model checkpoint file selection
st.sidebar.header("Model Selection")
ckpt_files = get_available_checkpoints("checkpoints")

if not ckpt_files:
    st.sidebar.error("No checkpoint (.pth) files found in the checkpoints/ directory.")
    st.stop()

ckpt_choice = st.sidebar.selectbox("Select a model checkpoint", ckpt_files)
checkpoint_path = os.path.join("checkpoints", ckpt_choice)
device_option = st.sidebar.selectbox("Inference device", options=["cpu", "cuda"], index=0)

model = load_model(checkpoint_path, device_option)
st.sidebar.success(f"Model '{ckpt_choice}' loaded successfully!")

# Image input
img_file = st.file_uploader("Upload an input image", type=["jpg", "jpeg", "png"])
if img_file:
    st.image(img_file, caption="Input Image", use_column_width=True)
    pil_img = Image.open(img_file)

    if st.button("Run 3D Reconstruction"):
        image_tensor = preprocess_image(pil_img)
        raw_voxels, binary_voxels = predict(model, image_tensor, device_option)

        st.subheader("3D Voxel Reconstruction")
        plot_fig = plot_voxels(binary_voxels, threshold=0.5, opacity=0.7)
        st.plotly_chart(plot_fig, use_container_width=True)

        st.write("Filled voxels:", int(np.sum(binary_voxels)))
        st.write("Voxel grid shape:", binary_voxels.shape)
else:
    st.info("Please upload an input image.")
