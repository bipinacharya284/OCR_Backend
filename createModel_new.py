# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your dataset
data = pd.read_csv('data.csv')

# Assuming your data.csv has 'label' column for target and rest are pixel values
# Prepare your data
labels = data['character']
pixels = data.drop('character', axis=1)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Normalize pixel values to be between 0 and 1
pixels = pixels / 255.0

# Split your data into training and testing sets
train_pixels, test_pixels, train_labels, test_labels = train_test_split(pixels, labels, test_size=0.2)

# Reshape your data to fit the model
train_pixels = np.array(train_pixels).reshape(-1, 32, 32, 1)
test_pixels = np.array(test_pixels).reshape(-1, 32, 32, 1)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False)  # Don't randomly flip images vertically

datagen.fit(train_pixels)

# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))) 
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))  

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

history = model.fit(datagen.flow(train_pixels, train_labels, batch_size=32),
                    epochs=50, 
                    validation_data=(test_pixels, test_labels),
                    callbacks=[early_stopping, learning_rate_reduction])

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
plt.savefig('accuracy_loss_plot.png')
plt.show()

model.save('ocr_model_new.h5')
