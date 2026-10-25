from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from data_preprocessing import get_data_generators
from model_architecture import build_model
from config import MODEL_SAVE_DIR
import os, glob

def load_latest_model(model_dir):
    """Load the most recent .h5 model from saved_models/ if available."""
    model_files = glob.glob(os.path.join(model_dir, "*.h5"))
    if not model_files:
        return None
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"✅ Resuming training from: {os.path.basename(latest_model)}")
    return load_model(latest_model)

def train_model(train_dir, epochs=40, batch_size=32, lr=5e-4):
    train_gen, val_gen = get_data_generators(train_dir, batch_size=batch_size)
    
    # Automatically get number of classes from data generator
    num_classes = len(train_gen.class_indices)
    print(f"📊 Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")

    # Try to resume from latest model
    model = load_latest_model(MODEL_SAVE_DIR)
    if model is None:
        print("⚡ No previous model found, starting fresh...")
        model = build_model(num_classes=num_classes)

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint_path = os.path.join(
        MODEL_SAVE_DIR, "best_model_epoch{epoch:02d}_valacc{val_accuracy:.2f}.h5"
    )
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )
    
    # Save training history to CSV
    csv_logger = CSVLogger(os.path.join(MODEL_SAVE_DIR, "training_history.csv"), append=True)

    history = None
    try:
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=[lr_scheduler, early_stopping, checkpoint, csv_logger]
        )
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user (Ctrl+C). Proceeding to evaluation...")

    return model, history
