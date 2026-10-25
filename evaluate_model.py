import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_preprocessing import get_data_generators
from config import TRAIN_DIR, IMG_SIZE, BATCH_SIZE, MODEL_SAVE_DIR
import os, glob

def evaluate_model(model=None, val_gen=None):
    if val_gen is None:
        _, val_gen = get_data_generators(TRAIN_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    if model is None:
        model_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.h5"))
        if not model_files:
            print("❌ No models found to evaluate.")
            return
        MODEL_PATH = max(model_files, key=os.path.getmtime)
        model = load_model(MODEL_PATH)
        print(f"✅ Loaded model: {MODEL_PATH}")

    # Evaluate model
    loss, acc = model.evaluate(val_gen)
    print(f"\n✅ Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    # Predictions
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen), axis=1)
    class_labels = list(val_gen.class_indices.keys())

    # Classification Report
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved to confusion_matrix.png")
    
    # Plot Training History from CSV
    csv_path = os.path.join(MODEL_SAVE_DIR, "training_history.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            plt.figure(figsize=(12, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(df['epoch'], df['accuracy'], label='Train Accuracy')
            plt.plot(df['epoch'], df['val_accuracy'], label='Val Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(df['epoch'], df['loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            print("✅ Training history graphs saved to training_history.png")
        except Exception as e:
            print(f"⚠️ Could not plot training history: {e}")
            
if __name__ == "__main__":
    evaluate_model()
