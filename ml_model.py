from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import load_dataset
import numpy as np
import joblib

def extract_features(images):
    """Extract simple features from images for traditional ML"""
    features = []
    for img in images:
        # Simple features: mean and std of each channel
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        features.append(np.concatenate([mean, std]))
    return np.array(features)

def train_ml_model(X_train, y_train, model_type='random_forest'):
    """Train a traditional ML model"""
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, random_state=42)
    else:
        raise ValueError("Unknown model type")
    
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load dataset
    images, labels, class_names = load_dataset("dataset/train")
    
    # Extract features
    X = extract_features(images)
    y = np.argmax(labels, axis=1)  # Convert from one-hot
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate models
    for model_type in ['random_forest', 'svm']:
        print(f"\nTraining {model_type}...")
        model = train_ml_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_type} accuracy: {accuracy:.2%}")
        
        # Save model
        joblib.dump(model, f"models/{model_type}_leaf_disease.joblib")