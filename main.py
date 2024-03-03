from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import io

from wordSegmentation import segment_word_into_characters


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Load the trained model
model = load_model('ocr_model_new.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

label_to_char = {
    'character_01_ka': 'क',
    'character_02_kha': 'ख',
    'character_03_ga': 'ग',
    'character_04_gha': 'घ',
    'character_05_kna': 'ङ',
    'character_06_cha': 'च',
    'character_07_chha': 'छ',
    'character_08_ja': 'ज',
    'character_09_jha': 'झ',
    'character_10_yna': 'ञ',
    'character_11_taamatar': 'ट',
    'character_12_thaa': 'ठ',
    'character_13_daa': 'ड',
    'character_14_dhaa': 'ढ',
    'character_15_adna': 'ण',
    'character_16_tabala': 'त',
    'character_17_tha': 'थ',
    'character_18_da': 'द',
    'character_19_dha': 'ध',
    'character_20_na': 'न',
    'character_21_pa': 'प',
    'character_22_pha': 'फ',
    'character_23_ba': 'ब',
    'character_24_bha': 'भ',
    'character_25_ma': 'म',
    'character_26_yaw': 'य',
    'character_27_ra': 'र',
    'character_28_la': 'ल',
    'character_29_waw': 'व',
    'character_30_motosaw': 'श',
    'character_31_petchiryakha': 'ष',
    'character_32_patalosaw': 'स',
    'character_33_ha': 'ह',
    'character_34_chhya': 'क्ष',
    'character_35_tra': 'त्र',
    'character_36_gya': 'ज्ञ',
    'digit_0': '०',
    'digit_1': '१',
    'digit_2': '२',
    'digit_3': '३',
    'digit_4': '४',
    'digit_5': '५',
    'digit_6': '६',
    'digit_7': '७',
    'digit_8': '८',
    'digit_9': '९',
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')

    # Save the image to a temporary file
    temp_file_path = 'temp.png'
    img.save(temp_file_path)

    # Segment the word into individual characters
    characters = segment_word_into_characters(temp_file_path)

    # Process each character
    predictions = []
    for char_img in characters:
        # Resize the image to 32x32 pixels
        char_img = char_img.resize((32, 32))

        # Invert the image
        char_img = ImageOps.invert(char_img)

        # Convert the image data to a numpy array and normalize it
        img_data = np.array(char_img) / 255.0

        # Reshape the data to the shape your model expects
        # For a single grayscale image, the shape is (1, 32, 32, 1)
        img_data = img_data.reshape(1, 32, 32, 1)

        # Use the model to make a prediction
        prediction = model.predict(img_data)

        # The prediction is an array of probabilities for each class
        # Use np.argmax to get the index of the highest probability
        predicted_class_index = np.argmax(prediction)

        # Use the label encoder to get the original class name
        predicted_class = encoder.inverse_transform([predicted_class_index])

        predicted_char = convert_label_to_text(predicted_class[0])

        predictions.append(predicted_char)

        prediction_string = ''.join(predictions)

    return {"predictions": prediction_string}


def convert_label_to_text(predicted_label):
    # Look up the actual character in the dictionary
    return label_to_char.get(predicted_label, "Unknown label")