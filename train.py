import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define dataset path
data_dir = r"C:\Users\user\OneDrive\Desktop\leaf-disease-detection\dataset\train"

# Define image parameters
img_size = (128, 128)
batch_size = 32

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')

val_generator = train_datagen.flow_from_directory(
    data_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')

# Fix input shape issue in the model
model = keras.Sequential([
    keras.Input(shape=(128, 128, 3)),  # Correct input shape
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save the model
model.save("leaf_disease_model.h5")

print("Model training complete and saved as 'leaf_disease_model.h5'")
