import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Set the path to your dataset
dataset_path = "dataset/Images/Images"

# Function to load and preprocess the dataset
def load_dataset(dataset_path):
    data = []
    labels = []
    
    character_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for label, character_dir in enumerate(character_dirs):
        character_path = os.path.join(dataset_path, character_dir)
        
        for filename in os.listdir(character_path):
            img_path = os.path.join(character_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32, 32))  # Adjust the size as needed
            img = img.astype('float32') / 255.0  # Normalize pixel values
            data.append(img)
            labels.append(label)
    
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels

# Load the dataset
X, y = load_dataset(dataset_path)

# Check if the dataset is empty
if len(X) == 0:
    print("Dataset is empty. Please check the path.")
    exit()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))  # Adjust the output size based on your number of characters

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape(-1, 32, 32, 1), y_train, epochs=10, validation_data=(X_test.reshape(-1, 32, 32, 1), y_test))

# Save the trained model
model.save('ocr_model.h5')
