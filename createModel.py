# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle


# Load your dataset
data = pd.read_csv('data.csv')

# Assuming your data.csv has 'label' column for target and rest are pixel values
# Prepare your data
labels = data['character']
pixels = data.drop('character', axis=1)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
# labels = labels.astype('int32')

# Normalize pixel values to be between 0 and 1
pixels = pixels / 255.0

# Split your data into training and testing sets
train_pixels, test_pixels, train_labels, test_labels = train_test_split(pixels, labels, test_size=0.2)

# Reshape your data to fit the model
train_pixels = np.array(train_pixels).reshape(-1, 32, 32, 1)
test_pixels = np.array(test_pixels).reshape(-1, 32, 32, 1)

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))

# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu')) # here 64 is the no of neurons
model.add(layers.Dense(len(np.unique(labels)), activation='sigmoid'))  # Number of labels are the 

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_pixels, train_labels, epochs=10, 
                    validation_data=(test_pixels, test_labels))


# Assuming 'encoder' is the LabelEncoder you used to encode your labels
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()




model.save('ocr_model.h5')