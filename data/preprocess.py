import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

def preprocess_image(image, img_size=(224, 224)):
    img = cv2.resize(image, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img
