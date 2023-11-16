import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers
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

'''Setting up human recognition dataset'''
# Define the paths for the true and false datasets
dataset_path = 'C:\Users\simos\Desktop\VAIHTO\KOULU\Intruder detection git\intruderdetection\human_dataset'

# Set the batch size
batch_size = 32

# Human dataset is with 3 channels (RGB), but the sizes vary. 
# We need to find perfect rescaled image size.
image_size = (256, 256)

print('Loading human dataset...')
human_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    batch_size=batch_size,
    label_mode='binary',  # Use binary labels (0 or 1)
    class_names=['1', '0'],  # Class names based on folder names
    image_size=image_size,  # Resize images to desired size
)

# Extract images and labels from the dataset
x = []
y = []

print('Converting tensorflow dataset to numpy...')
for images, labels in human_dataset:
    # .numpy() converts the images and labels objects from tf to numpy arrays
    x.append(images.numpy())
    y.append(labels.numpy())

# Concatenate batches into numpy arrays
x_human = np.concatenate(x)
y_human = np.concatenate(y)

# Split the dataset into training and testing sets
print('Performing train-test split...')
x_train_human, x_test_human, y_train_human, y_test_human = train_test_split(
    x_human, y_human, test_size=0.2, random_state=42, stratify=y_human
)

# Normalize the images
x_train_human = normalize_images(x_train_human)
x_test_human = normalize_images(x_test_human)

# Display the shapes of the resulting CIFAR sets
print("x_train_obj shape:", x_train_obj.shape)
print("y_train_obj shape:", y_train_obj.shape)
print("x_test_obj shape:", x_test_obj.shape)
print("y_test_obj shape:", y_test_obj.shape)

# Display the shapes of the resulting human sets
print("x_train_human shape:", x_train_human.shape)
print("y_train_human shape:", y_train_human.shape)
print("x_test_human shape:", x_test_human.shape)
print("y_test_human shape:", y_test_human.shape)
    
'''
# Define a simple CNN model with variable input size
model = tf.keras.Sequential([
    layers.Input(shape=(None, None, 3)),  # Input shape with variable height, width, and 3 channels
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification example
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''