# Plant_Disease_Detection

This repository contains a web application built using Streamlit that classifies plant diseases based on leaf images. The app utilizes a pre-trained deep learning model to make predictions on the uploaded images.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependenci Plant Disease Classifieres](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Plant diseases can significantly impact crop yield and quality, making it crucial to detect and identify them early. The Plant Disease Classifier aims to provide a user-friendly interface for farmers, gardeners, and anyone interested in identifying plant diseases by simply uploading a leaf image.

The application leverages a pre-trained deep learning model built using TensorFlow and Keras. The model was trained on the PlantVillage dataset, which contains images of various plant diseases across different plant species. The model training process is available in the `plant_disease_detection.py` file, which was originally created using Google Colab. You can find the Colab notebook at the following link:[https://colab.research.google.com/drive/18LPECKxoAMxMXybkSoq_0d05CU-S2jYe?usp=sharing]

## Features

- **User-friendly interface**: The app provides a clean and intuitive interface for users to upload leaf images and obtain disease predictions.
- **Image classification**: The pre-trained deep learning model accurately classifies plant diseases based on the uploaded leaf images.
- **Responsive design**: The app is designed to be responsive and accessible on various devices, including desktops, tablets, and smartphones.
- **Example image**: An example leaf image is provided in the sidebar for users to better understand the expected input format.

## Installation

To run the Plant Disease Classifier locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/plant-disease-detection.git
   ```

2. Navigate to the project directory:

   ```bash
   cd plant-disease-detection
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model file (`Plant_Disease_Detection.h5`) and the `class_indices.json` file, and place them in the `app/trained_models` directory.

## Usage

1. Start the Streamlit app:

   ```bash
   streamlit run app/main.py
   ```

2. The app will open in your default web browser. If not, you can navigate to the provided URL (e.g., `http://localhost:8501`).

3. In the app, click the "Choose an image..." button to upload a leaf image.

4. After selecting the image, click the "Classify" button to obtain the disease prediction.

5. The prediction will be displayed in the "Prediction" section.

## Project Structure

```
plant-disease-detection/
├── app/
│   ├── trained_models/
│   │   ├── Plant_Disease_Detection.h5
│   │   └── class_indices.json
│   ├── main.py
│   └── requirements.txt
├── notebook/
│   └── plant_disease_detection.py
└── README.md
```

- `app/`: Contains the Streamlit application files and the pre-trained model.
  - `trained_models/`: Directory for storing the pre-trained model and class indices.
  - `main.py`: The main Streamlit application file.
  - `requirements.txt`: List of required Python dependencies.
- `notebook/`: Contains the Python script used for training the deep learning model.
  - `plant_disease_detection.py`: Python script for training the model on the PlantVillage dataset.
- `README.md`: This file, providing an overview and instructions for the project.

## Dependencies

The main dependencies for this project are listed in the `requirements.txt` file. Here are the key dependencies:

- Streamlit: A Python library for building interactive web applications.
- TensorFlow: A popular open-source machine learning library for building and training deep learning models.
- Keras: A high-level neural networks API, used for building and training the deep learning model.
- Pillow: A Python library for image processing.
- NumPy: A library for scientific computing in Python.

## Contributing

Contributions to the Plant Disease Classifier project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).