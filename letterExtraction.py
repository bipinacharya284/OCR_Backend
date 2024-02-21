import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    return thresholded

def detect_contours(thresholded):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def segment_characters(contours, thresholded):
    characters = []

    for contour in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the character from the image
        character = thresholded[y:y+h, x:x+w]

        characters.append(character)

    return characters

def predict_characters(characters, model_path):
    # Load your pre-trained model
    model = load_model(model_path)

    predictions = []

    for character in characters:
        # Preprocess the character image and make a prediction
        character = cv2.resize(character, (32, 32))
        character = character.astype('float32') / 255
        character = np.expand_dims(character, axis=0)
        character = np.expand_dims(character, axis=-1)

        # Predict the character
        prediction = model.predict(character)

        predictions.append(prediction)

    return predictions

def main(image_path, model_path):
    thresholded = preprocess_image(image_path)
    contours = detect_contours(thresholded)
    characters = segment_characters(contours, thresholded)
    predictions = predict_characters(characters, model_path)

    return predictions

if __name__ == "__main__":
    image_path = 'image.png'  # Replace with your image path
    model_path = 'model.h5'  # Replace with your model path
    predictions = main(image_path, model_path)

    print(predictions)
