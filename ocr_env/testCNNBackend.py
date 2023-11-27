from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from starlette.responses import JSONResponse
from tensorflow import keras

app = FastAPI()

# Load the pre-trained model
model = keras.models.load_model('CNN_handwritten.model')

def preprocess_image(image):
    # Perform any necessary preprocessing on the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = np.invert(np.array([img]))
    img = img.astype('float32') / 255.0  # Normalize pixel values to be between 0 and 1
    return img

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        digit_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return JSONResponse(content={"predicted_digit": digit_class, "confidence": confidence})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
async def new():
    print("New")

