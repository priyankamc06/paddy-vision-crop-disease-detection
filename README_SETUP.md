# 🌾 Paddy Vision - Smart Crop Disease Detection System

AI-powered Paddy (Rice) crop disease diagnosis and health monitoring system using Deep Learning (InceptionV3 transfer learning), Flask, Streamlit, and FastAPI.

---

## 🚀 Quick Start (1-Click Run)

You can launch the project instantly using any of the following methods:

### 1. Desktop Shortcut
Double-click the **`Paddy Vision - Crop Disease Detection`** shortcut on your Desktop.

### 2. Batch Launchers (in `smart_crop_disease_detection\`)
- **`run_flask_app.bat`**: Launches the main web portal (Flask + AI model) on **http://127.0.0.1:5000** and automatically opens your browser.
- **`run_streamlit_app.bat`**: Launches the quick diagnostic Streamlit dashboard on **http://127.0.0.1:8501**.
- **`run_fastapi.bat`**: Launches the REST API with Swagger documentation on **http://127.0.0.1:8000/docs**.

---

## 🔑 Login & Access

- Open `http://127.0.0.1:5000` in your web browser.
- You can **Register a new account** directly using the "Register here" link with:
  - Username
  - 10-digit Phone Number (e.g. `9876543210`)
  - Place / City
  - Password
- Or log in with existing registered users in the local database.
- **Admin Dashboard**: Accessible at `http://127.0.0.1:5000/admin` to view registered farmers, total scans, detection history, and database schema.

---

## 🌿 Supported Disease Detections & Treatments

The InceptionV3 neural network model (`best_model_epoch02_valacc0.76.h5`) detects:
1. **Blast** (ಬೆಂಕಿ ರೋಗ / ब्लास्ट रोग)
2. **Healthy** (ಆರೋಗ್ಯಕರ / स्वस्थ)
3. **Insect Damage** (ಕೀಟ / कीट)
4. **Leaf Folder** (ಎಲೆ ಮಡಿಸುವ ಕೀಟ / पत्ती लपेटक)
5. **Tungro** (ತುಂಗ್ರೊ / टुंग्रो)

Each prediction provides:
- Confidence score (%)
- Recommended chemical and bio-fertilizers
- Preventive field management practices
- Step-by-step application instructions
- Multilingual interface support: **English**, **Kannada (ಕನ್ನಡ)**, and **Hindi (हिन्दी)**.

---

## 🤖 Optional: Gemini AI Chatbot & Verification

An optional `.env` file has been created at:
`dataset\balanced_dataset\.env`

To activate the smart Gemini chatbot & leaf image pre-validation:
1. Obtain a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Edit `dataset\balanced_dataset\.env`:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
*(If left empty, the application still works completely offline with the local InceptionV3 deep learning model!)*

---

## 📁 Project Structure

```
smart_crop_disease_detection/
├── .venv/                              # Isolated Python 3.12 virtual environment
├── run_flask_app.bat                   # 1-Click Flask launcher
├── run_streamlit_app.bat               # 1-Click Streamlit launcher
├── run_fastapi.bat                     # 1-Click FastAPI launcher
├── README_SETUP.md                     # Documentation
└── dataset/
    └── balanced_dataset/
        ├── app_flask.py                # Main Flask full-stack web application
        ├── App.py                      # Streamlit UI
        ├── backend_api.py              # FastAPI REST endpoints
        ├── flask_paddy.db              # SQLite database (Users & scan history)
        ├── saved_models/               # Pre-trained deep learning .h5 models
        ├── static/                     # CSS, images, client-side JS & translations
        ├── templates/                  # HTML templates (index, dashboard, admin)
        ├── train/                      # Training dataset images
        └── test/                       # Testing dataset images
```
cd "C:\Users\priyanka m c\Documents\vao\smart_crop_disease_detection"
.\.venv\Scripts\Activate.ps1
cd .\dataset\balanced_dataset
python app_flask.py