import os
import cv2
import random
from tqdm import tqdm
import albumentations as A
import shutil

# ====== CONFIG ======
SOURCE_DIR = r"C:\Users\RAHUL GOWDA\Downloads\archive (4)"
OUTPUT_DIR = r"C:\Users\RAHUL GOWDA\OneDrive\Desktop\dataset\balanced_dataset"

TRAIN_TARGET = 1500
TEST_TARGET = 300

CLASSES = ["blast", "healthy", "insect", "leaf_folder", "strips", "tungro"]

# ====== AUGMENTATION ======
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2)
])

# ====== CREATE FOLDERS ======
def create_folders():
    print("📁 Creating folder structure...")
    for split in ["train", "test"]:
        for cls in CLASSES:
            path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(path, exist_ok=True)
            print("Created:", path)

# ====== GET IMAGES ======
def get_images(path):
    if not os.path.exists(path):
        return []
    return [f for f in os.listdir(path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ====== COPY DATA ======
def copy_dataset():
    print("\n📂 Copying dataset...\n")

    for split in ["train", "test"]:
        for cls in CLASSES:
            src = os.path.join(SOURCE_DIR, split, cls)
            dst = os.path.join(OUTPUT_DIR, split, cls)

            if not os.path.exists(src):
                print(f"⚠️ Missing source: {src}")
                continue

            images = get_images(src)

            for img in images:
                shutil.copy(os.path.join(src, img),
                            os.path.join(dst, img))

            print(f"{cls} ({split}) copied: {len(images)} images")

# ====== BALANCE ======
def balance(split, target):
    print(f"\n⚙️ Balancing {split.upper()}...\n")

    for cls in CLASSES:
        path = os.path.join(OUTPUT_DIR, split, cls)
        images = get_images(path)

        count = len(images)
        print(f"{cls}: {count}")

        # 🚨 EMPTY CLASS FIX
        if count == 0:
            print(f"❌ No images in '{cls}' → Skipping\n")
            continue

        # ===== DELETE EXTRA =====
        if count > target:
            remove_count = count - target
            to_remove = random.sample(images, remove_count)

            for img in to_remove:
                os.remove(os.path.join(path, img))

            print(f"🗑 Reduced to {target}")

        # ===== AUGMENT =====
        elif count < target:
            needed = target - count
            print(f"✨ Augmenting {needed} images")

            for i in tqdm(range(needed)):
                img_name = random.choice(images)
                img_path = os.path.join(path, img_name)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug = transform(image=image)["image"]

                new_name = f"aug_{i}_{img_name}"
                save_path = os.path.join(path, new_name)

                cv2.imwrite(save_path, cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

        else:
            print("✅ Already balanced")

        print()

# ====== MAIN ======
if __name__ == "__main__":
    create_folders()      # STEP 1
    copy_dataset()        # STEP 2

    balance("train", TRAIN_TARGET)   # STEP 3
    balance("test", TEST_TARGET)

    print("\n✅ DONE!")