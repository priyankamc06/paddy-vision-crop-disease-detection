# Paddy Vision - Smart Crop Disease Detection

A Flask web app that scans paddy (rice) leaf images and classifies them into 5 classes: Blast, Healthy, Insect Damage, Leaf Folder, and Tungro.

## Features
- Image-based paddy leaf disease classification
- InceptionV3 transfer learning model
- 3-language dashboard (English, Kannada, Hindi)
- Admin module with SQLite storage

## Tech Stack
Python, Flask, TensorFlow/Keras, InceptionV3, SQLite, HTML, CSS, JavaScript

## Results
- 76.3% accuracy on 1,500 test images (5 classes x 300)

## How to Run
1. Install the requirements: `pip install -r requirements.txt`
2. Run: `python app_flask.py`
3. Open http://127.0.0.1:5000 in your browser

## Note
The trained model file is not included because of its size.

## Author
Priyanka M C# paddy-vision-crop-disease-detection
Flask web app that classifies 5 paddy leaf conditions using InceptionV3
