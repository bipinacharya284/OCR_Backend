import os
import pandas as pd
from PIL import Image
import numpy as np
import random
from tqdm import tqdm

# Get a list of all character directories
characters = os.listdir('dataset/images/images')

# Define the filename
filename = 'data_new_1.csv'

# Loop over all character directories
for character in tqdm(characters, desc="Processing characters"):
    # Initialize lists to store image data and labels for each character
    data = []
    labels = []

    # Get a list of all images in each character directory
    images = os.listdir(f'dataset/images/images/{character}')
    
    # Loop over all images in each character directory
    for image_file in tqdm(images, desc=f"Processing images for character {character}", leave=False):
        # Open image file
        with Image.open(f'dataset/images/images/{character}/{image_file}') as img:
            
            # Convert image to grayscale
            img = img.convert('L')

            # Convert image data to a flat list
            img_data = np.array(img).flatten().tolist()
            # Append image data and label to respective lists
            data.append(img_data)
            labels.append(character)

            # Generate a random degree between 1 and 30
            degree = random.randint(1, 30)

            # Perform image augmentation by rotating the image by a random degree clockwise
            img_rotated_clockwise = img.rotate(-degree)
            img_data_rotated_clockwise = np.array(img_rotated_clockwise).flatten().tolist()
            data.append(img_data_rotated_clockwise)
            labels.append(character)

            # Perform image augmentation by rotating the image by a random degree counterclockwise
            img_rotated_counterclockwise = img.rotate(degree)
            img_data_rotated_counterclockwise = np.array(img_rotated_counterclockwise).flatten().tolist()
            data.append(img_data_rotated_counterclockwise)
            labels.append(character)

    # Create a DataFrame from the image data
    df = pd.DataFrame(data)

    # Add labels as a new column to the DataFrame
    df['label'] = labels

    # Check if the file exists
    if os.path.isfile(filename):
        # If the file exists, append the data without the header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If the file does not exist, write the data and create the file
        df.to_csv(filename, index=False)
