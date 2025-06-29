# utils.py

import numpy as np
from tensorflow.keras.preprocessing import image

def load_labels(label_file='labels.txt'):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)