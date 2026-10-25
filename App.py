import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import glob

st.set_page_config(page_title="🌾 MediScan - Plant Disease Diagnosis")

MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
model_files = glob.glob(os.path.join(MODEL_SAVE_DIR, "*.h5"))

if not model_files:
    st.error(f"⚠️ No models found in '{MODEL_SAVE_DIR}'. Please train the model first by running `main.py`.")
    st.stop()

# Load the latest model
MODEL_PATH = max(model_files, key=os.path.getmtime)
model = load_model(MODEL_PATH)
st.success(f"✅ Loaded model: {os.path.basename(MODEL_PATH)}")

# Class labels for rice diseases based on the dataset folders
class_labels = ['blast', 'healthy', 'insect', 'leaf_folder', 'tungro']

# Function to preprocess and predict the image
def predict_image(uploaded_file):
    uploaded_image = Image.open(uploaded_file).convert("RGB")  # Force 3 channels
    uploaded_image = uploaded_image.resize((299, 299))         # Resize for InceptionV3
    image_array = image.img_to_array(uploaded_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class, confidence


# Streamlit UI
st.title("🌾 MediScan - AI Powered Plant Disease Diagnosis")
st.write("Upload a leaf image to detect rice diseases")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing the image...")
    predicted_class, confidence = predict_image(uploaded_file)
    st.success(f"Prediction: {class_labels[predicted_class]}")
    st.info(f"Confidence: {confidence:.2f}")
