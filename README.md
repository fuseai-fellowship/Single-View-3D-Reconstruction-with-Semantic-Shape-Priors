# Single-View-3D-Reconstruction-with-Semantic-Shape-Priors

## Setup instructions
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
2. Install pytorch package
    ```shell
    # If you have NVIDIA-GPU (CUDA 12.6)
    uv sync --extra cu126

    # If you dont
    uv sync --extra cpu
    ```
3. Install other required packages
    ```shell
    uv pip install -r requirements.txt
    ```

## Setup Dataset
```shell
mkdir ShapeNet/
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
wget http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
tar -xzf ShapeNetRendering.tgz -C ShapeNet/
tar -xzf ShapeNetVox32.tgz -C ShapeNet/

## See inside ShapeNet/ShapeNetRendering
# You will see a tarball, delete that tarball

## See inside ShapeNet/ShapeNetVox32
# You will see a tarball, delete that tarball
```
