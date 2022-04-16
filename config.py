import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET = 'anime'  # or 'map'
ROOT_DIR = '/content/data/data/train'
VAL_DIR = '/content/data/data/val'
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"