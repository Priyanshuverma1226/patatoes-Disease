import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

MODEL = tf.keras.models.load_model('models/model_1.keras')
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def preprocess_image(image):

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title('Patato Disease Classification App')
  
  
    

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            img_batch = np.expand_dims(image, 0)
            predictions = MODEL.predict(img_batch)
            
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            print(predicted_class)
            
            st.write('Prediction:', predictions)
            st.write('Prediction Class:', predicted_class)

if __name__ == '__main__':
    main()
