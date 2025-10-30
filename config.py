# --- Model Architecture ---
# Image Encoder (Section 3.3)
from torchvision import transforms

IMG_SIZE = 224
IMG_CHANNELS = 3
MAX_VIEWS = 1
INPUT_IMG_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
ENCODER_FEATURE_DIM = 256  # Output of the Image Encoder (Section 3.3)

# Memory Module (Section 3.1 & 4.2)
MEMORY_SIZE = 1000  # m (Section 4.2)
TOP_K_MEMORY = 3  # k (Hyperparameter for how many shapes to retrieve)

# LSTM Shape Encoder (Section 3.2 & 3.3)
LSTM_INPUT_DIM = 32768  # 32*32*32 (flattened voxel grid)
LSTM_HIDDEN_DIM = 2048  # (Section 3.3)
LSTM_NUM_LAYERS = 1  # (Section 3.3)

# Shape Decoder (Section 3.3)
# Input dim is image_feature + shape_prior_vector
DECODER_INPUT_DIM = ENCODER_FEATURE_DIM + LSTM_HIDDEN_DIM
VOXEL_RES = 32  # Final 32x32x32 output
DECODER_PROB_THRESHOLD = 0.5  # To binarize the sigmoid activation

# --- Training ---
# Train - Val - Test Split
TRAIN_SPLIT = [0, 0.7]
VAL_SPLIT = [0.7, 0.9]
TEST_SPLIT = [0.9, 1.0]
SAVE_PERIOD = 5

# Optimizer (Section 4.2)
LEARNING_RATE = 0.001  # 1e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
LR_DECAY_EPOCH = 150
LR_DECAY_FACTOR = 0.5  # Paper says "decayed by 2"

# Loss Functions (Section 4.2)
LOSS_LAMBDA = 1.0  # Weight for reconstruction loss (hyperparameter)
TRIPLET_MARGIN = 0.1  # alpha (Section 4.2)
TRIPLET_BETA = 0.85  # beta (Section 4.2)
MEMORY_DELTA = 0.90  # delta (Section 4.2)

# --- Runtime ---
DEVICE = "cuda"  # "cuda" or "cpu"
BATCH_SIZE = 32  # (Section 4.2)
NUM_EPOCHS = 300
NUM_WORKERS = 2  # Number of parallel workers for data loading
DATA_DIR = "./ShapeNet"
MODEL_SAVE_PATH = "./checkpoints/"
LOG_DIR = "./logs/"


# import argparse

# def get_args():

#     parser = argparse.ArgumentParser(description='Mem3D: Single-View 3D Reconstruction with Shape Priors')

#     # --- Paths ---
#     parser.add_argument('--data_path', type=str, default='/path/to/shapenet',
#                         help='Path to the root ShapeNet dataset (which contains 3D-R2N2 renderings)')
#     parser.add_argument('--log_dir', type=str, default='logs',
#                         help='Directory to save logs and checkpoints')
#     parser.add_argument('--split_file', type=str, default='data/train_val_test_split.json',
#                         help='Path to the json file defining train/val/test splits')

#     # --- Training Hyperparameters (from paper) ---
#     parser.add_argument('--epochs', type=int, default=200, # Paper mentions decay at 150, so total is likely >150
#                         help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Batch size for training')
#     parser.add_argument('--lr', type=float, default=1e-3, # 0.001
#                         help='Initial learning rate')
#     parser.add_argument('--num_workers', type=int, default=4,
#                         help='Number of workers for data loading')
#     parser.add_argument('--save_period', type=int, default=5,
#                         help='Save checkpoint every N epochs')

#     # --- Model Hyperparameters (from paper) ---
#     parser.add_argument('--feature_dim', type=int, default=256,
#                         help='Dimension of the image features (default, can be tuned)')
#     parser.add_argument('--memory_size', type=int, default=4000,
#                         help='Number of items in the memory network (m = 4000)')
#     parser.add_argument('--vox_dim', type=int, default=32,
#                         help='Resolution of the voxel grid (e.g., 32 for 32x32x32)')
#     parser.add_argument('--img_size', type=int, default=224,
#                         help='Size to resize input images to (224x224 in paper)')

#     # --- Loss & Memory Hyperparameters (from paper) ---
#     parser.add_argument('--delta', type=float, default=0.90,
#                         help='Similarity threshold (delta) for the Memory Writer (Sv)')
#     parser.add_argument('--beta', type=float, default=0.85,
#                         help='Similarity threshold (beta) for Triplet Loss negative selection (Sv)')
#     parser.add_argument('--margin', type=float, default=0.1,
#                         help='Margin (alpha) for the Voxel Triplet Loss (Sk)')
#     parser.add_argument('--lambda_recon', type=float, default=1.0,
#                         help='Weight for the reconstruction loss (default, can be tuned)')

#     # Use parse_known_args() to avoid errors in environments like Colab/Jupyter
#     args, _ = parser.parse_known_args()
#     return args
