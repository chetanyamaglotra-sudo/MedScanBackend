import tensorflow as tf
import numpy as np
import cv2
import json

def load_class_mapping(class_map_path):
    """Load the class label mapping saved after training."""
    with open(class_map_path, "r") as f:
        return json.load(f)

def classify_image(model_path, class_map_path, image_path, img_size=(224, 224)):
    """Load trained model and classify a single image."""
    
    model = tf.keras.models.load_model(model_path)


    class_indices = load_class_mapping(class_map_path)
    inv_class_map = {v: k for k, v in class_indices.items()}

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_resized = cv2.resize(image, img_size)
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = int(np.argmax(preds))
    top_conf = float(preds[top_idx])
    top_class = inv_class_map[top_idx]


    # print(f"✅ Prediction: {top_class} ({top_conf*100:.2f}% confidence)")
    # return [{"class": top_class, "confidence": top_conf}]
    return (f"✅ {top_class} ({top_conf*100:.2f}% confidence)")



