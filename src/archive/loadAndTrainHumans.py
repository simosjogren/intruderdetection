import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Function to normalize images to the range [0, 1]
def normalize_images(images):
    return images.astype("float32") / 255.0

'''Setting up human recognition dataset'''
# Define the paths for the true and false datasets
dataset_path = 'C://Users//simos//Desktop//VAIHTO//KOULU//Intruder detection git//intruderdetection//human_dataset//'

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
    color_mode='grayscale',  # Specify the color mode of the images
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

'''
# Select custom amount of pics for classification
num_images_per_class = 100
indices_class_0 = np.where(y_human == 0)[0][:num_images_per_class]
indices_class_1 = np.where(y_human == 1)[0][:num_images_per_class]
selected_indices = np.concatenate([indices_class_0, indices_class_1])
x_human_selected = x_human[selected_indices]
y_human_selected = y_human[selected_indices]
'''

# Split the selected dataset into training and testing sets
print('Performing train-test split for the selected dataset...')
x_train_human, x_test_human, y_train_human, y_test_human = train_test_split(
    x_human, y_human, test_size=0.2, random_state=42, stratify=y_human
)

# Normalize the images
x_train_human = normalize_images(x_train_human)
x_test_human = normalize_images(x_test_human)

# Display the shapes of the resulting human sets
print("x_train_human shape:", x_train_human.shape)
print("y_train_human shape:", y_train_human.shape)
print("x_test_human shape:", x_test_human.shape)
print("y_test_human shape:", y_test_human.shape)

# Define the model
model = models.Sequential()

# Feature extraction layers (CNN)
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the feature maps and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))  # Optional dropout layer for regularization

# Output layer for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Assuming x_train is your training data and y_train is your labels
model.fit(x_train_human, y_train_human, epochs=10, batch_size=32, validation_split=0.2)

# Display the model summary
model.summary()

# Lets save the model
model.save("../neural_networks/human_model.h5")

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_human, np.concatenate([y_test_human, np.zeros_like(y_test_human)], axis=1))
print("Test Accuracy:", test_acc)