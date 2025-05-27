# 🌿 Leaf Disease Detection using CNN

This project is a deep learning model built using **TensorFlow and Keras** to detect and classify different types of leaf diseases. It helps in early detection of crop issues by analyzing leaf images and identifying whether the leaf is healthy or affected by a disease.

---

## 🧠 Project Highlights

- Built with **Convolutional Neural Networks (CNN)**
- Uses **ImageDataGenerator** for data augmentation and preprocessing
- Trained on a custom dataset of plant leaf images
- Classifies leaves as:
  - `healthy`
  - `diseased_type1`
  - `diseased_type2`
- Real-time prediction using OpenCV and Matplotlib

---

## 🗂 Dataset

- The dataset should be organized like this:

dataset/
└── train/
├── healthy/
├── diseased_type1/
└── diseased_type2/


- You can use any plant leaf dataset by organizing it into class-labeled subdirectories.
- Images are resized to `128x128` pixels during training.

---

## ⚙️ Installation

1. **Clone this repo**

git clone https://github.com/your-username/leaf-disease-detection.git
cd leaf-disease-detection

🏗️ Model Architecture
The CNN model is built using Keras.Sequential API:

Input layer (128x128x3)

Conv2D → MaxPooling

Conv2D → MaxPooling

Flatten → Dense → Softmax

The model is trained for 10 epochs and saved as leaf_disease_model.h5.

python predict.py  # to predict the new test image
