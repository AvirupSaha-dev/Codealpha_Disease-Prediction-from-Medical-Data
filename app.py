import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MediPredict",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    heart_model = joblib.load("models/heart_disease_model.pkl")
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    breast_model = joblib.load("models/breast_cancer_model.pkl")

    return heart_model, diabetes_model, breast_model


heart_model, diabetes_model, breast_model = load_models()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🩺 MediPredict")

st.sidebar.caption("Medical Data Classification System")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Disease Prediction",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI/ML-based disease risk prediction using structured patient data."
)


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title("🩺 MediPredict")

    st.subheader("Disease Prediction from Medical Data")

    st.write(
        """
        MediPredict is a Machine Learning based web application that
        predicts the possibility of selected diseases using structured
        patient medical data.
        """
    )

    st.markdown("---")

    # Main project cards
    st.header("🎯 Supported Diseases")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ❤️ Heart Disease")
        st.write(
            "Predict heart disease risk using cardiovascular "
            "and clinical parameters."
        )

    with col2:
        st.markdown("### 🩸 Diabetes")
        st.write(
            "Predict diabetes risk using glucose, BMI, age "
            "and other patient parameters."
        )

    with col3:
        st.markdown("### 🔬 Breast Cancer")
        st.write(
            "Predict malignant-risk classification using "
            "cell nucleus measurements."
        )

    st.markdown("---")

    # ML Algorithms
    st.header("🤖 Machine Learning Algorithms")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("**Logistic Regression**")

    with col2:
        st.info("**Support Vector Machine**")

    with col3:
        st.info("**Random Forest**")

    with col4:
        st.info("**XGBoost**")

    st.markdown("---")

    # Selected models
    st.header("🏆 Selected Prediction Models")

    model_data = pd.DataFrame({
        "Disease": [
            "Heart Disease",
            "Diabetes",
            "Breast Cancer"
        ],
        "Selected Model": [
            "Random Forest",
            "XGBoost",
            "SVM"
        ],
        "F1 Score": [
            "0.9000",
            "0.6408",
            "0.9630"
        ]
    })

    st.dataframe(
        model_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.header("⚙️ How It Works")

    st.markdown(
        """
        **Step 1 — Enter Patient Data**

        The user manually enters the required medical parameters.

        **Step 2 — Data Processing**

        The entered values are converted into the numerical format
        expected by the trained model.

        **Step 3 — Machine Learning Prediction**

        The appropriate trained classification model analyzes the data.

        **Step 4 — Result**

        The system displays the predicted class and model-generated
        probability.
        """
    )

    st.markdown("---")

    st.success(
        "👉 Select **Disease Prediction** from the sidebar to begin."
    )


# =========================================================
# DISEASE PREDICTION
# =========================================================

elif page == "🔍 Disease Prediction":

    st.title("🔍 Disease Prediction")

    st.write(
        "Enter the patient's medical information and submit it "
        "to generate an ML-based prediction."
    )

    disease = st.selectbox(
        "Select Disease",
        [
            "❤️ Heart Disease",
            "🩸 Diabetes",
            "🔬 Breast Cancer"
        ]
    )

    st.markdown("---")


    # =====================================================
    # HEART DISEASE
    # =====================================================

    if disease == "❤️ Heart Disease":

        st.header("❤️ Heart Disease Prediction")

        st.caption(
            "Enter the patient's cardiovascular parameters."
        )

        with st.form("heart_form"):

            st.subheader("👤 Patient Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input(
                    "Age",
                    1,
                    120,
                    50
                )

            with col2:
                sex = st.selectbox(
                    "Sex",
                    ["Male", "Female"]
                )

            with col3:
                cp = st.selectbox(
                    "Chest Pain Type",
                    [
                        "Typical Angina",
                        "Atypical Angina",
                        "Non-anginal Pain",
                        "Asymptomatic"
                    ]
                )

            st.subheader("🫀 Cardiovascular Measurements")

            col1, col2, col3 = st.columns(3)

            with col1:
                trestbps = st.number_input(
                    "Resting Blood Pressure (mm Hg)",
                    50,
                    250,
                    120
                )

            with col2:
                chol = st.number_input(
                    "Serum Cholesterol (mg/dl)",
                    50,
                    700,
                    200
                )

            with col3:
                thalach = st.number_input(
                    "Maximum Heart Rate",
                    50,
                    250,
                    150
                )

            st.subheader("🧪 Additional Parameters")

            col1, col2, col3 = st.columns(3)

            with col1:
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl",
                    ["No", "Yes"]
                )

            with col2:
                restecg = st.selectbox(
                    "Resting ECG",
                    [
                        "Normal",
                        "ST-T Wave Abnormality",
                        "Left Ventricular Hypertrophy"
                    ]
                )

            with col3:
                exang = st.selectbox(
                    "Exercise Induced Angina",
                    ["No", "Yes"]
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                oldpeak = st.number_input(
                    "ST Depression (Oldpeak)",
                    0.0,
                    10.0,
                    1.0,
                    0.1
                )

            with col2:
                slope = st.selectbox(
                    "Slope",
                    [
                        "Upsloping",
                        "Flat",
                        "Downsloping"
                    ]
                )

            with col3:
                ca = st.number_input(
                    "Major Vessels (0–3)",
                    0,
                    3,
                    0
                )

            thal = st.selectbox(
                "Thalassemia",
                [
                    "Normal",
                    "Fixed Defect",
                    "Reversible Defect"
                ]
            )

            submitted = st.form_submit_button(
                "🔍 Predict Heart Disease",
                use_container_width=True
            )

        if submitted:

            cp_map = {
                "Typical Angina": 1,
                "Atypical Angina": 2,
                "Non-anginal Pain": 3,
                "Asymptomatic": 4
            }

            restecg_map = {
                "Normal": 0,
                "ST-T Wave Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }

            slope_map = {
                "Upsloping": 1,
                "Flat": 2,
                "Downsloping": 3
            }

            thal_map = {
                "Normal": 3,
                "Fixed Defect": 6,
                "Reversible Defect": 7
            }

            patient = [[
                age,
                1 if sex == "Male" else 0,
                cp_map[cp],
                trestbps,
                chol,
                1 if fbs == "Yes" else 0,
                restecg_map[restecg],
                thalach,
                1 if exang == "Yes" else 0,
                oldpeak,
                slope_map[slope],
                ca,
                thal_map[thal]
            ]]

            prediction = heart_model.predict(patient)[0]
            probability = heart_model.predict_proba(patient)[0][1]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error(
                    "⚠️ **Elevated Heart Disease Risk Predicted**"
                )
            else:
                st.success(
                    "✅ **No Elevated Heart Disease Risk Predicted**"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model Risk Score",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    "Random Forest"
                )


    # =====================================================
    # DIABETES
    # =====================================================

    elif disease == "🩸 Diabetes":

        st.header("🩸 Diabetes Prediction")

        st.caption(
            "Enter the patient's metabolic and clinical parameters."
        )

        with st.form("diabetes_form"):

            col1, col2 = st.columns(2)

            with col1:

                pregnancies = st.number_input(
                    "Pregnancies",
                    0,
                    20,
                    1
                )

                glucose = st.number_input(
                    "Glucose (mg/dL)",
                    1.0,
                    400.0,
                    120.0
                )

                blood_pressure = st.number_input(
                    "Blood Pressure (mm Hg)",
                    1.0,
                    250.0,
                    70.0
                )

                skin_thickness = st.number_input(
                    "Skin Thickness (mm)",
                    1.0,
                    100.0,
                    20.0
                )

            with col2:

                insulin = st.number_input(
                    "Insulin (μU/mL)",
                    1.0,
                    1000.0,
                    80.0
                )

                bmi = st.number_input(
                    "BMI",
                    1.0,
                    70.0,
                    25.0
                )

                diabetes_pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    0.0,
                    3.0,
                    0.5,
                    0.01
                )

                age = st.number_input(
                    "Age",
                    1,
                    120,
                    30
                )

            submitted = st.form_submit_button(
                "🔍 Predict Diabetes",
                use_container_width=True
            )

        if submitted:

            patient = [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]]

            prediction = diabetes_model.predict(patient)[0]
            probability = diabetes_model.predict_proba(patient)[0][1]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error(
                    "⚠️ **Elevated Diabetes Risk Predicted**"
                )
            else:
                st.success(
                    "✅ **No Elevated Diabetes Risk Predicted**"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model Risk Score",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    "XGBoost"
                )


    # =====================================================
    # BREAST CANCER
    # =====================================================

    else:

        st.header("🔬 Breast Cancer Prediction")

        st.caption(
            "Enter the cell-nucleus measurements required by the model."
        )

        st.info(
            """
            These are the 30 numerical features used by the
            trained breast cancer classification model.
            """
        )

        features = [
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

        with st.form("breast_form"):

            values = []

            st.subheader("🔬 Mean Features")

            for i in range(10):

                col1, col2, col3 = st.columns(3)

                index = i * 3

                with col1:
                    value = st.number_input(
                        features[index],
                        value=0.0,
                        format="%.5f",
                        key=f"breast_{index}"
                    )
                    values.append(value)

                with col2:
                    value = st.number_input(
                        features[index + 1],
                        value=0.0,
                        format="%.5f",
                        key=f"breast_{index + 1}"
                    )
                    values.append(value)

                with col3:
                    value = st.number_input(
                        features[index + 2],
                        value=0.0,
                        format="%.5f",
                        key=f"breast_{index + 2}"
                    )
                    values.append(value)

            st.subheader("⚠️ Note")

            st.write(
                "The breast cancer model requires measurements "
                "obtained from diagnostic cell-nucleus analysis."
            )

            submitted = st.form_submit_button(
                "🔍 Predict Breast Cancer Risk",
                use_container_width=True
            )

        if submitted:

            patient = pd.DataFrame(
                [values],
                columns=features
            )

            prediction = breast_model.predict(patient)[0]
            probability = breast_model.predict_proba(patient)[0][1]

            st.markdown("---")
            st.subheader("📊 Prediction Result")

            if prediction == 1:
                st.error(
                    "⚠️ **Predicted Malignant Risk**"
                )
            else:
                st.success(
                    "✅ **No Malignant Risk Predicted**"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Model Risk Score",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Model Used",
                    "SVM"
                )


# =========================================================
# ABOUT
# =========================================================

else:

    st.title("ℹ️ About MediPredict")

    st.header("📌 Project Objective")

    st.write(
        """
        The objective of MediPredict is to predict the possibility
        of selected diseases based on structured patient data using
        Machine Learning classification techniques.
        """
    )

    st.markdown("---")

    st.header("📚 Datasets")

    st.markdown(
        """
        - ❤️ Heart Disease Dataset
        - 🩸 Diabetes Dataset
        - 🔬 Breast Cancer Dataset
        """
    )

    st.markdown("---")

    st.header("🤖 Algorithms")

    st.markdown(
        """
        1. Logistic Regression
        2. Support Vector Machine (SVM)
        3. Random Forest
        4. XGBoost
        """
    )

    st.markdown("---")

    st.header("🔄 System Workflow")

    st.code(
        """
Patient enters medical data
          ↓
Data preprocessing
          ↓
Trained ML classification model
          ↓
Disease prediction
          ↓
Model risk score
        """,
        language="text"
    )

    st.markdown("---")

    st.header("⚠️ Disclaimer")

    st.warning(
        """
        MediPredict is an educational and research project.
        The predictions are generated by machine learning models
        trained on public datasets. They are not medical diagnoses
        and should not replace evaluation by a qualified healthcare
        professional.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🩺 MediPredict | Machine Learning Based Disease Prediction"
)