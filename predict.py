import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

# Parameters
MODEL_PATH = r"C:\Users\user\OneDrive\Desktop\leaf-disease-detection\leaf_disease_model.h5"
IMG_SIZE = (128, 128)

# Load model
model = load_model(MODEL_PATH)

# Class names (update these based on your dataset)
CLASS_NAMES = ['healthy', 'diseased_type1', 'diseased_type2']

def predict_image(image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error: Could not read image"
    
    display_img = img.copy()
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = CLASS_NAMES[class_idx]
    
    return display_img, f"Prediction: {class_name} ({confidence:.2%})"

# Example usage
if __name__ == "__main__":
    image_path = input("Enter path to leaf image: ").strip()
    if not os.path.exists(image_path):
        print("Error: File not found")
    else:
        img, result = predict_image(image_path)
        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(result)
            plt.axis('off')
            plt.show()
            print(result)