import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path):
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

    # Apply a Gaussian blur to reduce noise 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding 
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2) 

    # Add morphological operations here to remove small noise - use opening 
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
   
    final_img = cv2.bitwise_not(opening) 

    preprocessed_img = Image.fromarray(final_img)
    preprocessed_img.save('word_segmentation\\preprocessed_image.png')

# Test with an example file path; replace with actual file path when running 
preprocess_image('word_segmentation\\kalama_nocam.jpg')
