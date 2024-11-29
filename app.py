import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    # Add your class mappings
    classes = {
        0: 'Speed Limit 20 km/h',
        1: 'Speed Limit 30 km/h',

     2:  'Speed Limit 50 km/h',
     3:  'Speed Limit 60 km/h',
     4:  'Speed Limit 70 km/h',
     5:  'Speed Limit 80 km/h',
     6:  'End of Speed Limit 80 km/h',
     7:  'Speed Limit 100 km/h',
     8:  'Speed Limit 120 km/h',
     9:  'No passing',
     10:  'No passing for vechiles over 3.5 metric tons',
     11:  'Right-of-way at the next intersection',
     12:  'Priority road',
     13:  'Yield',
     14:  'Stop',
     15:  'No vechiles',
     16:  'Vechiles over 3.5 metric tons prohibited',
     17:  'No entry',
     18:  'General caution',
     19:  'Dangerous curve to the left',
     20:  'Dangerous curve to the right',
     21:  'Double curve',
     22:  'Bumpy road',
     23:  'Slippery road',
     24:  'Road narrows on the right',
     25:  'Road work',
     26:  'Traffic signals',
     27:  'Pedestrians',
     28:  'Children crossing',
     29:  'Bicycles crossing',
     30:  'Beware of ice/snow',
     31:  'Wild animals crossing',
     32:  'End of all speed and passing limits',
     33:  'Turn right ahead',
     34:  'Turn left ahead',
     35:  'Ahead only',
     36:  'Go straight or right',
     37:  'Go straight or left',
     38:  'Keep right',
     39:  'Keep left',
        40:  'Roundabout mandatory',
        41:  'End of no passing',
        42: 'End of no passing by vehicles over 3.5 metric tons'
    }
    return classes.get(classNo, "Unknown")

def model_predict(image, model):
    # Preprocess the image
    img = cv2.resize(image, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)  
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)  # Get the predicted class index
    return getClassName(classIndex[0])           # Return the class name

# Add custom styles
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #ff4b4b;
        text-align: center;
    }
    .sidebar {
        background-color: #ffefef;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown("<h1 class='title'>Traffic Sign Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload an image to classify the traffic sign.</p>", unsafe_allow_html=True)
    
    st.sidebar.header("About")
    st.sidebar.info("""
        **Traffic Sign Recognition App**
        This app uses a pre-trained deep learning model to classify traffic signs.
        Upload an image, and the app will predict the sign's meaning.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        st.markdown("<p style='text-align:center; font-weight:bold;'>Classifying...</p>", unsafe_allow_html=True)
        
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        if image_np.shape[-1] == 4:  # Handle RGBA images
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        
        result = model_predict(image_np, model)
        st.markdown(f"<h3 style='text-align:center; color:green;'>Prediction: {result}</h3>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <hr>
        <p style='text-align:center;'>
       7th Semester Major project 
       
        </p>
         <p style='text-align:center;'>
       21052915 </p>
        <p style='text-align:center;'>
       21052919 </p>
       
       <p style='text-align:center;'>
       21052900 </p>
       <p style='text-align:center;'>
       21052813 </p>
       <p style='text-align:center;'>
       21052842 </p>
       <p style='text-align:center;'>
       21052961 </p>
       
        
        """, 
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
