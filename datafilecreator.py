import os
import pandas as pd
from PIL import Image
import numpy as np

# Initialize lists to store image data and labels
data = []
labels = []

# Loop over all character directories
for character in os.listdir('dataset/images/images'):
    # Loop over all images in each character directory
    for image_file in os.listdir(f'dataset/images/images/{character}'):
        # Open image file
        with Image.open(f'dataset/images/images/{character}/{image_file}') as img:
            # Convert image data to a flat list
            img_data = np.array(img).flatten().tolist()
            # Append image data and label to respective lists
            data.append(img_data)
            labels.append(character)

# # Create a DataFrame from the image data
df = pd.DataFrame(data)

# # Add labels as a new column to the DataFrame
df['label'] = labels

# # Write DataFrame to a CSV file
df.to_csv('data.csv', index=False)
