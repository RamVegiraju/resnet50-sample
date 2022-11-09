import os
import tensorflow as tf
from tensorflow.keras.applications import resnet50
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import backend
import numpy as np
from tensorflow.keras.preprocessing import image

print("test")


#model = tf.keras.applications.ResNet50()
tf.keras.backend.set_learning_phase(0)
model = resnet50.ResNet50()

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("kitten.jpg", target_size=(224, 224))
# Convert the image to a numpy array
x = image.img_to_array(img)
# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Scale the input image to the range used in the trained network
x = resnet50.preprocess_input(x)

print("predicting model")
predictions = model.predict(x)
predicted_classes = resnet50.decode_predictions(predictions, top=9)
print(predicted_classes)
