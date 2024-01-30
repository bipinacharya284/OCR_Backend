from fastapi import FastAPI, UploadFile, File
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import io

app = FastAPI()

# Load the trained model
model = load_model('ocr_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('L')

    # Resize the image to 32x32 pixels
    img = img.resize((32, 32))

    # Invert the image
    img = ImageOps.invert(img)

    # Convert the image data to a numpy array and normalize it
    img_data = np.array(img) / 255.0

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

    return {"prediction": predicted_class[0]}
