# 🗑️ Garbage Classification using Deep Learning

This project uses a Convolutional Neural Network (CNN) based on **MobileNetV2** to classify images of garbage into one of six categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

It is designed to help in automatic waste classification for efficient recycling and environmental awareness.

---

## 🚀 Quick Start – Test It in Google Colab

You can test the model directly in Colab without any setup.

### ▶️ Run the test notebook:
👉 [Open in Google Colab](https://colab.research.google.com/drive/1kvN44DJIqWYZxneyLmKLf3UBFTe1DXUR#scrollTo=yIbZ2cq0nNox)

### 📥 Download the model:
👉 [Download Model File (.h5)](https://drive.google.com/file/d/1gVKApkV8a6BNIqUnpsqYDJUcgP3cZ5pA/view?usp=drive_link)

---

### 🧠 How It Works

1. The model is trained using MobileNetV2 with fine-tuning
2. It expects RGB images resized to **224×224**
3. Users upload the model and any test image
4. The notebook classifies the image and shows:
   - Predicted label (e.g., plastic)
   - Confidence score
   - Input image preview

---

## 📁 Project Structure
garbage-classifier/
├── Garbage_Classifier.ipynb # Full training notebook (optional)
├── Garbage_Classifier_TestOnly.ipynb # Ready-to-run test-only notebook ✅
├── garbage_classifier_mobilenetv2.h5 # Trained model
├── sample_image.jpg # Example image (optional)
├── README.md # This file
├── requirements.txt # List of dependencies


---

## 🛠️ Tech Stack

- Google Colab
- TensorFlow + Keras
- MobileNetV2 (Transfer Learning)
- Python (PIL, NumPy, Matplotlib)

---


