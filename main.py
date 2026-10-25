from train_model import train_model
from evaluate_model import evaluate_model
from data_preprocessing import get_data_generators
from config import TRAIN_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE

try:
    # Train the model (will resume if previous best model exists)
    model, history = train_model(TRAIN_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE)
except KeyboardInterrupt:
    print("\n🛑 Execution interrupted by user! Skipping to evaluation step...")

# Evaluate on validation set
_, val_gen = get_data_generators(TRAIN_DIR, batch_size=BATCH_SIZE)
evaluate_model(val_gen=val_gen)
