# necessary library imported
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Loading dataset 
data = pd.read_csv('data.csv')

# Seperating the image pixels with the label 
labels = data['character']
pixels = data.drop('character', axis=1)

encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Normalizing the pixel value 
pixels = pixels / 255.0

# Spliting data into training and testing sets
train_pixels, test_pixels, train_labels, test_labels = train_test_split(pixels, labels, test_size=0.2)

# Reshaping data to fit the model
train_pixels = np.array(train_pixels).reshape(-1, 32, 32, 1)
test_pixels = np.array(test_pixels).reshape(-1, 32, 32, 1)

# Augumenting data
datagen = ImageDataGenerator(
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=False)  # Don't randomly flip images vertically

datagen.fit(train_pixels)

# Creating convolutional base layer
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())

# Adding Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))) 
model.add(Dropout(0.5))
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))  

# Compilining and training the model
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

# Saving the entire model to HDF5 file
model.save('ocr_model_new.h5')

# Ploting training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Ploting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Predicting the values from the validation dataset
y_pred = model.predict(test_pixels)
# Converting predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred, axis = 1) 

# Convert the labels back to original names 
test_labels_names = encoder.inverse_transform(test_labels)
y_pred_classes_names = encoder.inverse_transform(y_pred_classes)

# Computing the confusion matrix
confusion_mtx = confusion_matrix(test_labels_names, y_pred_classes_names) 

# Ploting the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')  # Save the confusion matrix to a file
plt.show()

# Preparing Classification report
print('Classification Report')
classification_rep = classification_report(test_labels_names, y_pred_classes_names)
print(classification_rep)

# Saving the classification report to a file
with open('classification_report.txt', 'w') as f:
    f.write(classification_rep)
