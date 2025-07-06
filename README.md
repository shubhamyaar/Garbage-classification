# 🗑️ Garbage Classification using Deep Learning (MobileNetV2)

This project is a Garbage Classification model built with **TensorFlow** and **MobileNetV2**, trained on 6 classes:  
**Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, and **Trash**.

🧠 The model achieves high validation accuracy and is designed to work on real-world images to predict garbage type efficiently.

---

## 📁 Project Structure


---

## 🚀 Getting Started

### ▶️ Run on Google Colab

👉 **[Open in Google Colab](https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID)**

> Replace `YOUR_NOTEBOOK_ID` with your actual Colab notebook share link ID.

---

## 📥 Model File

The model file `garbage_classifier.h5` is too large to host directly on GitHub.

📎 Download it from Google Drive:  
👉 [Download Model from Drive](https://drive.google.com/file/d/YOUR_MODEL_ID/view?usp=sharing)

> After downloading, place it in the same directory as your notebook (`garbage_classifier_notebook.ipynb`) or upload it inside Colab.

---

## 🖼️ Predicting a Custom Image

In the notebook:
1. Upload your image (JPEG/PNG) using the upload button.
2. The model will preprocess and classify the image.
3. Output: **Predicted Class + Confidence**

---

## 📊 Training Info

- **Base Model**: MobileNetV2 (pretrained on ImageNet, fine-tuned)
- **Epochs**: 20  
- **Final Val Accuracy**: ~87.5%
- **Loss**: 0.37

---

## 🛠️ Technologies Used

- Python 🐍
- TensorFlow / Keras 🧠
- Google Colab 💻
- Pillow / Matplotlib 📊

---

## 🙌 Contributing

Pull requests and suggestions are welcome!

---

## 📜 License

MIT License – feel free to use, modify, and distribute.
