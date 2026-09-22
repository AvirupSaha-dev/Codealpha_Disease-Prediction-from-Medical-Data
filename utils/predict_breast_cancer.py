import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/breast_cancer_model.pkl")


def predict_breast_cancer(patient_data):

    columns = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension",
        "radius error",
        "texture error",
        "perimeter error",
        "area error",
        "smoothness error",
        "compactness error",
        "concavity error",
        "concave points error",
        "symmetry error",
        "fractal dimension error",
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst compactness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension"
    ]

    data = pd.DataFrame([patient_data], columns=columns)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Malignant Cancer Risk Detected"
    else:
        result = "Benign / No Malignant Risk Detected"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "result": result
    }


# Test patient
patient = [
    14.0,      # mean radius
    20.0,      # mean texture
    90.0,      # mean perimeter
    600.0,     # mean area
    0.10,      # mean smoothness
    0.10,      # mean compactness
    0.08,      # mean concavity
    0.04,      # mean concave points
    0.18,      # mean symmetry
    0.06,      # mean fractal dimension
    0.30,      # radius error
    1.0,       # texture error
    2.0,       # perimeter error
    30.0,      # area error
    0.006,     # smoothness error
    0.02,      # compactness error
    0.03,      # concavity error
    0.01,      # concave points error
    0.02,      # symmetry error
    0.003,     # fractal dimension error
    16.0,      # worst radius
    25.0,      # worst texture
    105.0,     # worst perimeter
    800.0,     # worst area
    0.13,      # worst smoothness
    0.25,      # worst compactness
    0.30,      # worst concavity
    0.12,      # worst concave points
    0.30,      # worst symmetry
    0.08       # worst fractal dimension
]

result = predict_breast_cancer(patient)

print("\n========== BREAST CANCER PREDICTION ==========")
print("Prediction:", result["result"])
print("Risk Probability:", f"{result['probability'] * 100:.2f}%")