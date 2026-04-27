# config.py
import os

# Paths (adjust if needed)
DATA_ROOT = "nih_images"          # folder containing images (recursively searched)
LABEL_FILE = "Data_Entry_2017.csv"

# Model
BACKBONE = "densenet121"
NUM_CLASSES = 5                    # Pneumonia, Effusion, Atelectasis, Cardiomegaly, Infiltration
IMG_SIZE = 224
DROPOUT = 0.2

# Training
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 7
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Data augmentation
USE_CLACHE = True
USE_NOISE = True

# Device (will be set at runtime)
DEVICE = "cuda"