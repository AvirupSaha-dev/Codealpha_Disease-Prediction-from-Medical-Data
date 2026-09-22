# 🩺 MediPredict – Disease Prediction from Medical Data

MediPredict is a machine learning-based web application that predicts the possibility of selected diseases using structured patient medical data.

## 🚀 Live Demo

🔗 **Deployed Application:**  
https://codealphadisease-prediction-from-medical-data-h4knzrpwcxfdnz6e.streamlit.app/

> The live link will be updated after deployment.

---

## 🎯 Project Objective

The objective of this project is to apply machine learning classification techniques to medical datasets and predict the possibility of diseases based on patient-related features such as age, symptoms, blood pressure, glucose level, cholesterol, and other medical measurements.

The project currently supports:

- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction
- 🎗️ Breast Cancer Prediction

---

## 🤖 Machine Learning Algorithms

The project experiments with:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

The best-performing model based on F1-score was selected for each disease.

| Disease | Selected Model |
|---|---|
| Heart Disease | Random Forest |
| Diabetes | XGBoost |
| Breast Cancer | SVM |

---

## 📊 Datasets

The project uses publicly available medical datasets:

- UCI Heart Disease Dataset
- Pima Indians Diabetes Dataset
- Breast Cancer Dataset

The datasets are cleaned and processed before model training.

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Matplotlib
- Seaborn

---

## 🔄 Workflow

```text
Medical Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Processing
      ↓
Train Multiple ML Models
      ↓
Evaluate Models
      ↓
Select Best Model
      ↓
Save Trained Model
      ↓
Streamlit Web Application
      ↓
Disease Risk Prediction