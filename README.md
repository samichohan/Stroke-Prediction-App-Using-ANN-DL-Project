# Stroke-Prediction-App-Using-ANN-DL-Project
Stroke Prediction using ANN (TensorFlow) with Streamlit Web App | Deep Learning Healthcare Project with Threshold Tuning &amp; Model Evaluation
# 🧠 Stroke Prediction using ANN (TensorFlow)

## 📌 Project Overview
This project builds a Deep Learning model using Artificial Neural Networks (ANN) to predict whether a person is at risk of stroke based on medical and lifestyle data.

The model is trained on a real-world healthcare dataset and demonstrates how AI can assist in early disease detection.

---

## 🚀 Features
- Data preprocessing (handling missing values, encoding, scaling)
- ANN model built using TensorFlow/Keras
- Performance evaluation (Accuracy, Precision, Recall, F1 Score)
- Threshold tuning for better recall
- Real-time prediction on new patient data
- Visualization of training & validation performance

---

## 📊 Dataset Information
The dataset includes features such as:
- Age
- Gender
- Hypertension
- Heart Disease
- BMI
- Glucose Level
- Smoking Status
- Work Type
- Residence Type

**Target Variable:**
- `stroke` → 0 (No Stroke), 1 (Stroke)

---

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## 🧠 Model Architecture
- Input Layer
- Dense (64 neurons, ReLU)
- Dense (32 neurons, ReLU)
- Dense (16 neurons, ReLU)
- Dense (8 neurons, ReLU)
- Output Layer (Sigmoid)

---

## 📈 Model Performance

| Metric     | Score |
|-----------|------|
| Accuracy  | ~90% |
| Precision | ~0.28 |
| Recall    | ~0.38 |
| F1 Score  | ~0.32 |

---

## ⚠️ Important Insight
The dataset is imbalanced, so accuracy alone is not reliable.

To improve detection of stroke cases:
- Threshold tuning was applied (0.5 → 0.1)
- Recall improved significantly

---

## 📊 Training Graphs

### Accuracy Graph
![Accuracy](images/accuracy.png)

### Loss Graph
![Loss](images/loss.png)

---

## 🔮 Real-Time Prediction

The model can predict stroke risk for new patient data:

- Input: Patient features
- Output: Probability of stroke
- Decision based on threshold (0.1)

---

## 📦 Installation

```bash
pip install -r requirements.txt
