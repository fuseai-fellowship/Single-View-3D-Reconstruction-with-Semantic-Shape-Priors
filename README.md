# Single-View-3D-Reconstruction-with-Semantic-Shape-Priors

## Setup Environment
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

## Training Script
```shell
## Training hyperparameters
## Change values in config.py
# BATCH_SIZE
# NUM_EPOCHS

## Necessary paths
## Change values in config.py
# DATA_DIR: Path to dataset
# MODEL_SAVE_PATH
# LOG_DIR

uv run python main.py
```

## Inference Script
```shell
## DECODER_PROB_THRESHOLD to binarize the voxels
python inference.py \
    --image_path /path/to/image \
    --checkpoint_path /path/to/model \
    --device cuda/cpu \
    --save_visualization out_viz.png \
    --save_voxels out_voxels.npy

## Example
python inference.py \
    --image_path /home/apil/work/fuse-3d/ShapeNet/ShapeNetRendering/04379243/1a00aa6b75362cc5b324368d54a7416f/rendering/02.png \
    --checkpoint_path /home/apil/work/fuse-3d/checkpoints/best_model.pth \
    --device cuda \

```

### Results

    - Voxel grid: The dimensions of the 3D voxel grid
    - Filled voxels: Percentage of total voxels that are filled
    - Avg similarity: The average cosine similarity between the input image and the top-K retrieved shapes from memory