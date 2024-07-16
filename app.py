import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Function to download and load the model
model = load_model("./model.h5")

# Define the image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize image
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Normalize the image
    image = image / 255.0
    
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)  # Expand dimensions if necessary
    
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define the Streamlit app
st.title("😷 COVID-19 X-ray Analyzer")
st.write("Upload an X-ray image and get the classification result (COVID-19, Non-COVID, Normal).")

# File uploader
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the image resized to 512x512
    display_image = cv2.resize(image, (512, 512))
    
    # Preprocess the image for prediction
    processed_image = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = ['COVID-19', 'Non-COVID', 'Normal']
    
    # Get the highest probability and its corresponding label
    highest_prob = np.max(predictions[0])
    highest_prob_label = class_labels[predicted_class]
    
    # Display diagnostic message
    st.write(f"The diagnostic is **{highest_prob_label}** with **{highest_prob*100:.2f}%** probability.")
    
    # Display the uploaded image
    st.image(display_image, caption='Uploaded X-ray image.', use_column_width=True)


