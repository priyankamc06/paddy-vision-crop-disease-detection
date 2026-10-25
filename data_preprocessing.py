from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(train_dir, img_size=(299, 299), batch_size=32, val_split=0.2):
    train_path = os.path.join(train_dir, "train")
    # Changed from 'val' to 'test' to match your dataset structure
    val_path = os.path.join(train_dir, "test")

    if os.path.exists(val_path):  
        print("📂 Found test/ folder. Using explicit train/test split.")
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            directory=train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )

        validation_generator = val_datagen.flow_from_directory(
            directory=val_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    else:
        print("⚠️ No test/ folder found. Splitting train/ into train+val automatically.")
        datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=val_split
        )

        train_generator = datagen.flow_from_directory(
            directory=train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        validation_generator = datagen.flow_from_directory(
            directory=train_path,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

    return train_generator, validation_generator
