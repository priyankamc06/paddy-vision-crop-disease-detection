import os

# Use current directory as the dataset root
TRAIN_DIR = "."

# Image settings
IMG_SIZE = (299, 299)   # Required size for InceptionV3
BATCH_SIZE = 32
EPOCHS = 40             # More epochs for fine-tuning
LEARNING_RATE = 5e-4    # Will auto-reduce when plateau

# Folder to save best models
MODEL_SAVE_DIR = "saved_models"

# Auto-create folder if missing
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
