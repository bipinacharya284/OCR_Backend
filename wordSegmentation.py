import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def segment_word_into_characters(image_path):
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Check if the image is grayscale
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # The image is already grayscale
        gray = img_array
    else:
        # The image is not grayscale, convert it
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Threshold the image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segment the word into individual characters
    characters = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Adjust the bounding box to include the "dika"
        dika_offset = h // 4  # Adjust this value based on your specific requirements
        y = max(0, y - dika_offset)
        h = min(img.height - y, h + dika_offset)

        # Check the area of the bounding box
        if w * h > 100:  # Adjust this value based on your specific requirements
            # Crop the image to the bounding box
            char_img = img.crop((x, y, x + w, y + h))
            characters.append(char_img)

        # Draw the bounding box for visualization
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.imshow(img_array, cmap='gray')
    plt.show()

    return characters

# # Test the function with an image file
# characters = segment_word_into_characters('word_segmentation\\kalama.jpg')
# for i, char_img in enumerate(characters):
#     char_img.save(f'word_segmentation\\character_{i}.png')
