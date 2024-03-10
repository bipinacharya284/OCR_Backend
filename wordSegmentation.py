import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def segment_word_into_characters(image_path):
    # Opening the image file
    img = Image.open(image_path)

    # Converting the image to a numpy array
    img_array = np.array(img)

    # Checking if the image is grayscale
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # If image is already grayscale
        gray = img_array
    else:
        # If the image is not grayscale, convert it
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Thresholding the image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Finding contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segmenting the word into individual characters
    characters = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Adjusting the bounding box to include the "dika"
        dika_offset = h // 4  # Asjustment can be done based on requirements
        y = max(2, y - dika_offset)
        h = min(img.height - y, h + dika_offset)

        # Checking the area of the bounding box
        if w * h > 100:  # Adjustment can be done based on area of character
            # Croping the image to the bounding box
            char_img = img.crop((x, y, x + w, y + h))
            characters.append(char_img)

        # Drawing the bounding box for visualization
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Displaying the image with bounding boxes
    plt.imshow(img_array, cmap='gray')
    plt.show()

    return characters