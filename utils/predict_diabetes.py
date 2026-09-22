import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/diabetes_model.pkl")


def predict_diabetes(patient_data):

    columns = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ]

    data = pd.DataFrame([patient_data], columns=columns)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Diabetes Risk Detected"
    else:
        result = "No Diabetes Risk Detected"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "result": result
    }


# Test patient
patient = [
    2,      # Pregnancies
    120,    # Glucose
    70,     # BloodPressure
    20,     # SkinThickness
    79,     # Insulin
    25.0,   # BMI
    0.3,    # DiabetesPedigreeFunction
    30      # Age
]

result = predict_diabetes(patient)

print("\n========== DIABETES PREDICTION ==========")
print("Prediction:", result["result"])
print("Risk Probability:", f"{result['probability'] * 100:.2f}%")