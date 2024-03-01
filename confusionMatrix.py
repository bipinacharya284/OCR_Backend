# Import necessary libraries
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle

# Load your saved model
model = load_model('ocr_model_new.h5')

# Assuming you have two test images for two characters
# Load your test images and preprocess them in the same way as your training images
test_images = ['testData\\1358.png','testData\\8675.png','testData\\3734.png']
test_images = [Image.open(img).convert('L') for img in test_images]  # convert image to grayscale
test_images = [np.array(img.resize((32, 32))) for img in test_images]  # resize image to 32x32
test_images = [img.reshape(32, 32, 1) for img in test_images]  # reshape to match the model's expected input shape
test_images = np.array(test_images)

# Assuming the labels of the test images are 'character_01_ka' and 'character_03_ga'
test_labels = np.array(['character_01_ka','character_02_kha', 'character_03_ga'])  # replace with your test labels

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

# Transform the labels to the encoded labels
test_labels_encoded = encoder.transform(test_labels)

# Use the model to predict the test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Get the original labels from the encoded labels
predicted_labels_original = encoder.inverse_transform(predicted_labels)

# Generate the confusion matrix
labels = ['character_01_ka','character_02_kha', 'character_03_ga']  # specify the classes that are present in your test data
cm = confusion_matrix(test_labels, predicted_labels_original, labels=labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# Save the confusion matrix plot
plt.savefig('confusion_matrix.png')

plt.show()
