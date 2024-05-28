import json
import os.path
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# Set Streamlit page configuration as the first Streamlit command
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide"
)

# Load model and class indices
working_dir=os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/Trained_model/Plant_Disease_Detection.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Load and preprocess image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    image_batch = np.expand_dims(img_array, axis=0)
    img_array = image_batch.astype('float32') / 255.0
    return img_array

# Predict image class
def predict_image_class(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Custom CSS for better aesthetics and animations
st.markdown("""
    <style>
        body {
            background-color: #ffffff; 
            color: #000000;
        }
        .main {
            background-color: #000000; 
            font-family: 'Arial', sans-serif;
        }
        .button-container {
            display: flex;
            justify-content: center;
        }
        .upload-section, .result-section {
            background-color: #000000; 
            color: #ffff00;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, opacity 0.3s ease;
        }
        .upload-section.hidden, .result-section.hidden {
            transform: translateY(-20px);
            opacity: 0;
        }
        .upload-section.show, .result-section.show {
            transform: translateY(0);
            opacity: 1;
        }
        .uploaded-image img {
            border-radius: 10px;
            max-width: 100%;
        }
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const uploadSection = document.querySelector('.upload-section');
            const resultSection = document.querySelector('.result-section');
            if (uploadSection) {
                uploadSection.classList.remove('hidden');
                uploadSection.classList.add('show');
            }
            if (resultSection) {
                resultSection.classList.remove('hidden');
                resultSection.classList.add('show');
            }
        });
    </script>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>🌿 Plant Disease Classifier</h1></div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar with explanation and example image
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This web app classifies plant disease images into different categories. "
    "Upload an image of a plant leaf and click the **Classify** button to see the prediction."
)

st.sidebar.title("🌿 Example Image")
example_image = Image.open("/home/mukesh/Pictures/Screenshots/Screenshot from 2024-05-25 12-12-54.png")
st.sidebar.image(example_image, caption="Example Leaf", use_column_width=True)

# Main content area for image upload and classification
st.markdown('<div class="upload-section hidden"><h2>📋 Upload Image</h2></div>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    st.markdown('<div class="upload-section show"><h2>🖼️ Uploaded Image</h2></div>', unsafe_allow_html=True)
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format="auto")

    if st.button('Classify'):
        with st.spinner('🔍 Classifying...'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<div class="result-section show"><h2>🏆 Prediction: {prediction}</h2></div>', unsafe_allow_html=True)
