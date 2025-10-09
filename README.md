# Single-View-3D-Reconstruction-with-Semantic-Shape-Priors

## Setup instructions
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
2. Install required packages
    ```shell
    # If you have NVIDIA-GPU (CUDA 12.6)
    uv sync --extra cu126

    # If you dont
    uv sync --extra cpu
    ```