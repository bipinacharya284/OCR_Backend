from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import preprocessing

app = FastAPI()

# Load the trained model
model = load_model('ocr_model.h5')  # Use the correct path to your saved model

predicted_devanagaris = []

def process_image(file):
    # Load the single image
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    processed_img = preprocessing.preProcess(img)
    predicted_devanagaris.clear()
    
     # Mapping from label to Devanagari character
    label_to_devanagari = {
        31: 'क', 32: 'ख', 33: 'ग', 34: 'घ', 35: 'ङ', 36: 'च', 37: 'छ',
        38: 'ज', 39: 'झ', 40: 'ञ', 41: 'ट', 42: 'ठ', 43: 'ड', 44: 'ढ', 45: 'ण',
        46: 'त', 1: 'थ', 2: 'द', 3: 'ध', 4: 'न', 5: 'प', 6: 'फ', 7: 'ब', 8: 'भ',
        9: 'म', 10: 'य', 11: 'र', 12: 'ल', 13: 'व', 14: 'श', 15: 'ष', 16: 'स', 17: 'ह',
        18: 'क्ष', 19: 'त्र', 20: 'ज्ञ',
        # Numbers
        21: '०', 22: '१', 23: '२', 24: '३', 25: '४', 26: '५', 27: '६', 28: '७', 29: '८', 30: '९'
    }

    while not processed_img.empty():
    # Perform prediction
        prediction = model.predict(processed_img.get().reshape(1, 32, 32, 1))

        # Extract the label number directly from the prediction
        predicted_label = np.argmax(prediction) + 1  # Increase by 1 to match original labels



        # Convert predicted label to Devanagari character
        predicted_devanagari = label_to_devanagari.get(predicted_label, f'Unknown_{predicted_label}')
        print(predicted_devanagari)
        predicted_devanagaris.append(predicted_devanagari)
        # return predicted_devanagari
    

@app.post("/predict")
async def predict_text_endpoint(file: UploadFile = File(...)):
    try:
        process_image(file)
        return JSONResponse(content={"predicted_text": predicted_devanagaris}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
