# 🩺 Breast Cancer Detection App

A Machine Learning-powered web application built using **Streamlit**, **Scikit-learn**, and **TensorFlow** for early diagnosis of breast cancer from histopathological features. This app supports predictions using Random Forest, SVM, and Neural Networks, helping users understand malignancy risk from input features.

---

## 🔍 Overview

Breast cancer is one of the most common and life-threatening diseases among women worldwide. Early detection and diagnosis significantly improve treatment outcomes. This project leverages machine learning techniques to classify tumors as **benign** or **malignant** based on diagnostic features derived from digitized images of breast mass.

---

## 🚀 Features

- 📊 **Multiple ML Models**: Random Forest, SVM, and Neural Network
- 🔎 **Interactive Predictions** using manually entered or uploaded data
- 📈 **Model Evaluation**: ROC curves, confusion matrices, and accuracy metrics
- 💾 **Saved Model** for inference via `.pkl` files
- 🌐 **Streamlit Web Interface** for usability

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**
- **Scikit-learn**, **TensorFlow**, **Keras**
- **Streamlit** for the front-end
- **Joblib** for model serialization

---

## 📂 Dataset

- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Filename**: `Cancer_Data.csv`
- 30 features derived from images of cell nuclei + diagnosis label.

---

📈 Model Performance
| Model          | Accuracy |
| -------------- | -------- |
| Random Forest  |  ~98%    |
| SVM            |  ~97%    |
| Neural Network |  ~98%    |
