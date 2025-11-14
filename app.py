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

def create_cube_mesh(x, y, z, size=0.95):
    """Create vertices and faces for a single cube at position (x, y, z)"""
    # Define cube vertices (8 corners)
    s = size / 2
    vertices = np.array([
        [x-s, y-s, z-s], [x+s, y-s, z-s], [x+s, y+s, z-s], [x-s, y+s, z-s],  # bottom
        [x-s, y-s, z+s], [x+s, y-s, z+s], [x+s, y+s, z+s], [x-s, y+s, z+s]   # top
    ])
    
    # Define faces (12 triangles, 2 per face)
    faces = np.array([
        [0,1,2], [0,2,3],  # bottom
        [4,5,6], [4,6,7],  # top
        [0,1,5], [0,5,4],  # front
        [2,3,7], [2,7,6],  # back
        [0,3,7], [0,7,4],  # left
        [1,2,6], [1,6,5]   # right
    ])
    
    return vertices, faces

def plot_voxels_mesh(voxel_grid, threshold=0.5, colorscale='Viridis', title="3D Voxel Reconstruction"):
    """
    Create 3D voxel visualization using mesh representation with individual cubes.
    Provides better visual quality and clearer voxel boundaries.
    """
    # Find occupied voxels
    if voxel_grid.dtype == bool:
        occupied = voxel_grid
        values = voxel_grid.astype(float)
    else:
        occupied = voxel_grid > threshold
        values = voxel_grid
    
    indices = np.argwhere(occupied)
    
    if len(indices) == 0:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No voxels above threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
    else:
        # Collect all vertices and faces
        all_vertices = []
        all_faces = []
        all_intensities = []
        vertex_offset = 0
        
        for idx in indices:
            x, y, z = idx
            vertices, faces = create_cube_mesh(x, y, z, size=0.95)
            
            all_vertices.append(vertices)
            all_faces.append(faces + vertex_offset)
            all_intensities.extend([values[x, y, z]] * len(vertices))
            
            vertex_offset += len(vertices)
        
        # Combine all meshes
        vertices = np.vstack(all_vertices)
        faces = np.vstack(all_faces)
        intensities = np.array(all_intensities)
        
        # Create mesh3d trace
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                intensity=intensities,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title="Voxel<br>Value",
                    len=0.7,
                    thickness=15
                ),
                lighting=dict(
                    ambient=0.5,
                    diffuse=0.9,
                    specular=0.6,
                    roughness=0.5,
                    fresnel=0.2
                ),
                flatshading=False,
                hovertemplate='Value: %{intensity:.3f}<extra></extra>'
            )
        ])
    
    nx, ny, nz = voxel_grid.shape
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(
                title="X",
                range=[-1, nx],
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.3)",
                backgroundcolor="rgb(20, 20, 20)"
            ),
            yaxis=dict(
                title="Y",
                range=[-1, ny],
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.3)",
                backgroundcolor="rgb(20, 20, 20)"
            ),
            zaxis=dict(
                title="Z",
                range=[-1, nz],
                showgrid=True,
                gridcolor="rgba(128, 128, 128, 0.3)",
                backgroundcolor="rgb(20, 20, 20)"
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor="rgb(20, 20, 20)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgb(15, 15, 15)",
        plot_bgcolor="rgb(20, 20, 20)",
        font=dict(color="white", size=12),
        height=600
    )
    
    return fig

def plot_voxels_scatter(voxel_grid, threshold=0.5, title="3D Voxel Reconstruction (Point Cloud)"):
    """
    Alternative visualization using 3D scatter plot - faster for large voxel grids.
    """
    if voxel_grid.dtype == bool:
        occupied = voxel_grid
        values = voxel_grid.astype(float)
    else:
        occupied = voxel_grid > threshold
        values = voxel_grid
    
    indices = np.argwhere(occupied)
    
    if len(indices) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No voxels above threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
    else:
        voxel_values = values[occupied]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=indices[:, 0],
                y=indices[:, 1],
                z=indices[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=voxel_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Voxel<br>Value", len=0.7, thickness=15),
                    line=dict(width=0.5, color='rgba(255, 255, 255, 0.3)')
                ),
                hovertemplate='Position: (%{x}, %{y}, %{z})<br>Value: %{marker.color:.3f}<extra></extra>'
            )
        ])
    
    nx, ny, nz = voxel_grid.shape
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title="X", range=[-1, nx], backgroundcolor="rgb(20, 20, 20)"),
            yaxis=dict(title="Y", range=[-1, ny], backgroundcolor="rgb(20, 20, 20)"),
            zaxis=dict(title="Z", range=[-1, nz], backgroundcolor="rgb(20, 20, 20)"),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor="rgb(20, 20, 20)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgb(15, 15, 15)",
        font=dict(color="white", size=12),
        height=600
    )
    
    return fig

# ------------- Streamlit App -------------

st.set_page_config(page_title="Mem3D Visualizer", page_icon="🧊", layout="wide")

st.title("🧊 Mem3D: Single-View 3D Reconstruction")
st.markdown("""
Select a trained model checkpoint, upload an image, and reconstruct its 3D voxel representation!
""")

# Sidebar configuration
st.sidebar.header("⚙️ Model Configuration")
ckpt_files = get_available_checkpoints("checkpoints")

if not ckpt_files:
    st.sidebar.error("No checkpoint (.pth) files found in the checkpoints/ directory.")
    st.stop()

ckpt_choice = st.sidebar.selectbox("📦 Model Checkpoint", ckpt_files)
checkpoint_path = os.path.join("checkpoints", ckpt_choice)
device_option = st.sidebar.selectbox("🖥️ Inference Device", options=["cpu", "cuda"], index=0)

# Visualization settings
st.sidebar.header("🎨 Visualization Settings")
viz_mode = st.sidebar.radio(
    "Rendering Mode",
    options=["Mesh (High Quality)", "Point Cloud (Fast)"],
    help="Mesh mode shows solid cubes, Point Cloud mode is faster for large grids"
)
colorscheme = st.sidebar.selectbox(
    "Color Scheme",
    options=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Blues', 'Turbo'],
    index=0
)

# Load model
with st.spinner('Loading model...'):
    model = load_model(checkpoint_path, device_option)
st.sidebar.success(f"✅ Model '{ckpt_choice}' loaded!")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Input Image")
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        pil_img = Image.open(img_file)
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("🎯 Reconstruction Controls")
    if img_file:
        threshold = st.slider(
            "Voxel Threshold",
            0.0, 1.0,
            float(cfg.DECODER_PROB_THRESHOLD),
            0.05,
            help="Only voxels with values above this threshold will be displayed"
        )
        
        run_button = st.button("🚀 Run 3D Reconstruction", type="primary", use_container_width=True)
    else:
        st.info("👆 Please upload an image to begin")
        run_button = False

# Reconstruction and visualization
if img_file and run_button:
    with st.spinner('🔄 Running 3D reconstruction...'):
        image_tensor = preprocess_image(pil_img)
        raw_voxels, binary_voxels = predict(model, image_tensor, device_option)
    
    st.success("✅ Reconstruction complete!")
    
    # Statistics
    filled_voxels = np.sum(binary_voxels)
    total_voxels = binary_voxels.size
    fill_ratio = filled_voxels / total_voxels
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📐 Voxel Grid Shape", str(binary_voxels.shape))
    with col2:
        st.metric("🧱 Filled Voxels", f"{filled_voxels:,} / {total_voxels:,}")
    with col3:
        st.metric("📊 Fill Ratio", f"{fill_ratio:.2%}")
    
    st.markdown("---")
    
    # Visualization
    st.subheader("🎭 3D Visualization")
    
    tab1, tab2 = st.tabs(["Raw Voxels", "Binary Voxels"])
    
    with tab1:
        st.markdown(f"**Probability values with threshold: {threshold:.2f}**")
        if viz_mode == "Mesh (High Quality)":
            raw_fig = plot_voxels_mesh(raw_voxels, threshold=threshold, 
                                      colorscale=colorscheme, title="Raw Voxel Probabilities")
        else:
            raw_fig = plot_voxels_scatter(raw_voxels, threshold=threshold,
                                         title="Raw Voxel Probabilities")
        st.plotly_chart(raw_fig, use_container_width=True)
    
    with tab2:
        st.markdown(f"**Binary mask with threshold: {cfg.DECODER_PROB_THRESHOLD:.2f}**")
        if viz_mode == "Mesh (High Quality)":
            binary_fig = plot_voxels_mesh(binary_voxels.astype(np.float32),
                                         threshold=0.0, colorscale=colorscheme,
                                         title="Binary Voxel Reconstruction")
        else:
            binary_fig = plot_voxels_scatter(binary_voxels.astype(np.float32),
                                            threshold=0.0,
                                            title="Binary Voxel Reconstruction")
        st.plotly_chart(binary_fig, use_container_width=True)
    
    st.info("💡 **Interactive Controls:** Rotate (drag), Zoom (scroll), Pan (shift+drag)")