import uvicorn
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

# Allow all origins (React frontend or public use)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.tflite")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "classes.json")
IMG_SIZE = (224, 224)

# Load class map
with open(CLASS_MAP_PATH) as f:
    class_map = json.load(f)

inv_class_map = {v: k for k, v in class_map.items()}

# Load TFLite model
print("🔄 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("✅ TFLite model ready")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def predict_tflite(image_bytes):
    # Convert bytes → NumPy image
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None, None

    img_resized = cv2.resize(img, IMG_SIZE)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    label = inv_class_map[top_idx]

    return label, confidence


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    label, confidence = predict_tflite(contents)

    if label is None:
        return {"error": "Invalid image"}

    return {
        "label": label,
        "confidence": round(confidence * 100, 2)
    }


@app.get("/")
def home():
    return {"status": "running", "message": "TFLite backend is live!"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
