import cv2
import numpy as np
from queue import Queue

extracts = []
cropped_images = []
final_images = Queue()

def detect_and_extract_word(img):
    # Read the input image
    # img = cv2.imread(image_path)

    # Convert the image to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Apply adaptive thresholding to obtain a binary image
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Use morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours in the cleaned image
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Iterate over contours and extract characters
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the character in the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract individual character
        char_img = gray[y:y+h, x:x+w]

        # extracts.append(char_img)
        extracts.append(char_img)
        # Save the character image
        # char_output_path = f"{output_directory}/character_{i}.png"
        # cv2.imwrite(char_output_path, char_img)
        
        cv2.imshow("Extracted Words: ",char_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(i)
        # return char_img

    # # Display the marked image
    # cv2.imshow("Marked Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def crop_top(img, crop_height):
    # Read the input image
    # img = cv2.imread(input_path)

    # Crop the specified portion from the top
    cropped_image = img[crop_height:, :]

    # Save the cropped image
    # cv2.imwrite(output_path, cropped_image)

    cropped_images.append(cropped_image)

    # Display the original and cropped images
    # cv2.imshow("Original Image", img)
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_horizontal_lines(img):
    black_line_height = 10
    white_line_height = 20

    # Get the width of the input image
    img_width = img.shape[1]

    # Create a black image with the specified black line height and width
    black_line = np.zeros((black_line_height, img_width, 3), dtype=np.uint8)

    # Create a white image with the specified white line height and width
    white_line = 255 * np.ones((white_line_height, img_width, 3), dtype=np.uint8)

    # Ensure img is a 3D array
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Resize black_line and white_line to match the height of img
    black_line_resized = cv2.resize(black_line, (img_width, black_line_height))
    white_line_resized = cv2.resize(white_line, (img_width, white_line_height))

    # Concatenate the white line, black line, and the original image
    result_image = np.concatenate((white_line_resized, black_line_resized, img), axis=0)

    # Resize the result image to the desired size
    result_image_resized = cv2.resize(result_image, (32, 32))

    # Convert to grayscale
    result_image_resized = cv2.cvtColor(result_image_resized, cv2.COLOR_BGR2GRAY)

    # Invert the image
    result_image_resized = 255 - result_image_resized

    # Normalize pixel values
    result_image_resized = result_image_resized.astype('float32') / 255.0

    final_images.put(result_image_resized)

    # result_image = cv2.resize(result_image, (32, 32))  # Adjust the size as needed
    # result_image = 255 - result_image
    # result_image = result_image.astype('float32') / 255.0  # Normalize pixel values

    # final_images.put(result_image)
    
    # Display the image with horizontal lines
    cv2.imshow("Image with Horizontal Lines", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
# input_image_path = "3.png"

def preProcess(img):
    # img = cv2.imread(input_image_path)
    # Detect and extract characters
    detect_and_extract_word(img)

    # print(extracted_words)

    while extracts:
        # Crop the top portion
        # cv2.imshow("Image with Horizontal Lines", extracts.get())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        crop_top(extracts.pop(), crop_height=20)

    # print(cropped_images)

    while cropped_images:
        detect_and_extract_word(cropped_images.pop())
        # cv2.imshow("Cropped Images", cropped_images.get())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    while extracts:
        # cv2.imshow("Seperate Images", extracts.pop())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        add_horizontal_lines(extracts.pop())

    return final_images


