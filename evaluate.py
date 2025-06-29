# evaluate.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

val_path = 'dataset/val'
img_size = (150, 150)
batch_size = 32

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

model = tf.keras.models.load_model('garbage_classifier_week2.h5')

loss, acc = model.evaluate(val_data)
print(f"Validation Accuracy: {acc * 100:.2f}%")