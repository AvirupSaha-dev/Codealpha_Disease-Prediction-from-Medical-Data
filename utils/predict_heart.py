import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/heart_disease_model.pkl")


def predict_heart_disease(patient_data):
    """
    patient_data must contain the 13 heart disease features
    in the same order as the training dataset.
    """

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    data = pd.DataFrame([patient_data], columns=columns)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Heart Disease Risk Detected"
    else:
        result = "No Heart Disease Risk Detected"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "result": result
    }


# Test patient
patient = [
    55,     # age
    1,      # sex
    1,      # cp
    140,    # trestbps
    240,    # chol
    0,      # fbs
    1,      # restecg
    150,    # thalach
    0,      # exang
    1.0,    # oldpeak
    1,      # slope
    0,      # ca
    3       # thal
]

result = predict_heart_disease(patient)

print("\n========== HEART DISEASE PREDICTION ==========")
print("Prediction:", result["result"])
print("Risk Probability:", f"{result['probability'] * 100:.2f}%")