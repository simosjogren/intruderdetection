import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

'''We need to get the ssl module and turn of the ssl verification to get the cifar dataset.'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Function to normalize images to the range [0, 1]
def normalize_images(images):
    return images.astype("float32") / 255.0

'''Setting up the cifar dataset for object recognition'''
print('Loading CIFAR10 dataset...')
# Cifar pictures are 32x32 pixels with 3 channels (RGB)
(x_train_obj, y_train_obj), (x_test_obj, y_test_obj) = cifar10.load_data()
y_train_obj_binary = (y_train_obj > 0).astype(int)
y_test_obj_binary = (y_test_obj > 0).astype(int)

# Normalize the images
x_train_obj = normalize_images(x_train_obj)
x_test_obj = normalize_images(x_test_obj)

# Display the shapes of the resulting CIFAR sets
print("x_train_obj shape:", x_train_obj.shape)
print("y_train_obj shape:", y_train_obj.shape)
print("x_test_obj shape:", x_test_obj.shape)
print("y_test_obj shape:", y_test_obj.shape)

