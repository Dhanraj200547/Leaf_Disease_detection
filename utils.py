import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_dataset(data_path, img_size=(128, 128)):
    """
    Load dataset from directory and preprocess images
    """
    images = []
    labels = []
    class_names = os.listdir(data_path)
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            try:
                # Read and preprocess image
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize pixel values
                
                images.append(img)
                labels.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    # Convert labels to one-hot encoding
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)
    
    return np.array(images), np.array(images), labels, class_names

def split_dataset(images, labels, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets
    """
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)