import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model
model = tf.keras.models.load_model('fashion_mnist_cnn_model.h5')

# Class names for Fashion MNIST
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Streamlit app
st.title("Fashion MNIST Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Show prediction
    st.write(f"Predicted class: {predicted_class}")
