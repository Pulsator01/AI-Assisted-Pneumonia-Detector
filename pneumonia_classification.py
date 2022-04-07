import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
import time
CATEGORIES = ["NORMAL", "PNEUMONIA"]
relpath = r"CNN Pneumonia/saved_model.pb"
def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)



model = tf.keras.models.load_model('CNN Pneumonia')
st.title('Hello this model aims at identifying pneumonia by looking at X-rays.')
uploaded_file = st.file_uploader(label="Please upload your X-Ray", type=["JPEG", "JPG", "PNG"])
if uploaded_file is not None:
    try:
        path = st.text_input('Add the path to your image without quotes')
        image = Image.open(uploaded_file)
        st.image(image, caption='This is your uploaded file')
        prediction = model.predict([prepare(path)])
        outcome = (CATEGORIES[int(prediction[0][0])])
        if outcome =='PNEUMONIA':
            st.write('PNEUMONIA DETECTED. Please consult a medical expert')
        elif outcome == 'NORMAL':
            st.write('NO ISSUE DETECTED. Lungs seem to be normal.')
    
    except Exception as e:
        time.sleep(10)
        st.write('There was an error. Please try again later or refresh the page.'
                ' Make sure to enter the data accurately and put the path WITHOUT quotes')
else:
    print()
