import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import tensorflow.keras.backend as K

# Define image dimensions and input shape
img_rows = 28 
img_cols = 28
input_shape = (img_rows, img_cols, 1)

# Build the model architecture and load weights (cached)
@st.cache_resource
def build_model():
    K.clear_session() # Clear the Keras session
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    model.load_weights("models/models.h5") # Load weights inside the cached function
    return model

# Function to preprocess the image
def preprocess_image(image_data):
    # Convert the image data from the canvas (RGBA) to grayscale (L)
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    return img

# Build and load weights into the model
loaded_model = build_model()

# Streamlit app layout
st.title('Handwritten Digit Recognition Web App')

col1, col2 = st.columns(2)

with col1:
    st.write("Draw a digit in the box below:")
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Add buttons below the canvas
    predict_button = st.button("Predict")
    clear_button = st.button("Clear")

with col2:
    st.write("Prediction")

    # Initialize state for prediction and confidence if not already present
    if 'predicted_digit' not in st.session_state:
        st.session_state.predicted_digit = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None

    # Display prediction and confidence if available
    if st.session_state.predicted_digit is not None:
        st.markdown(f"## {st.session_state.predicted_digit}")
        st.write(f"Confidence: {st.session_state.confidence:.2f}%")

# Handle button clicks
if predict_button:
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
        # Preprocess the image from the canvas
        processed_image = preprocess_image(canvas_result.image_data)

        # Predict the digit
        prediction = loaded_model.predict(processed_image)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Update session state
        st.session_state.predicted_digit = predicted_digit
        st.session_state.confidence = confidence

        # Rerun to update the display
        st.rerun()
    else:
        st.warning("Please draw a digit on the canvas.")

if clear_button:
    # Clearing the canvas requires rerunning the app with a potentially different key or state management
    # A simple way is to clear the prediction and rerun, the canvas might reset or can be explicitly handled
    st.session_state.predicted_digit = None
    st.session_state.confidence = None
    st.rerun() 