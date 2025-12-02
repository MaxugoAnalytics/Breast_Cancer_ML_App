import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Page configuration

st.set_page_config(
    page_title="Maxwell Adigwe Breast Cancer Prediction App",
    layout="wide",
    page_icon="Maxwell Adigwe Breast Cancer Prediction App 🎗️")

st.markdown("<h1 style='text-align: center; color: purple;'>🎗️ Breast Cancer Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the patient's cell measurements to predict if the tumor is Benign or Malignant.</p>", unsafe_allow_html=True)
st.markdown("---")

# Paths to models

model_dir = '.'  
scaler_path = os.path.join(model_dir, 'scaler.pkl')
logreg_model_path = os.path.join(model_dir, 'logistic_regression_model.joblib')
rf_model_path = os.path.join(model_dir, 'random_forest_model.joblib')
svm_model_path = os.path.join(model_dir, 'svm_model.joblib')

# Load models and scaler

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load(scaler_path)
        logreg_model = joblib.load(logreg_model_path)
        rf_model = joblib.load(rf_model_path)
        svm_model = joblib.load(svm_model_path)
        return scaler, logreg_model, rf_model, svm_model
    except FileNotFoundError as e:
        st.error(f"Error: Missing model file: {e.filename}")
        st.stop()

scaler, logreg_model, rf_model, svm_model = load_models()

model_dict = {
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "Support Vector Machine": svm_model
}

# Sidebar: Model selection
st.sidebar.header("Step 1: Select Model")
selected_model_name = st.sidebar.selectbox("Choose Prediction Model", list(model_dict.keys()))
model = model_dict[selected_model_name]


# Sidebar: Input features

st.sidebar.header("Step 2: Enter Tumor Features")

feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1', 'fractal_dimension1',
    'radius2', 'texture2', 'perimeter2', 'area2', 'smoothness2',
    'compactness2', 'concavity2', 'concave_points2', 'symmetry2', 'fractal_dimension2',
    'radius3', 'texture3', 'perimeter3', 'area3', 'smoothness3',
    'compactness3', 'concavity3', 'concave_points3', 'symmetry3', 'fractal_dimension3']

default_values = {
    'radius1': 17.99, 'texture1': 10.38, 'perimeter1': 122.8, 'area1': 1001.0,
    'smoothness1': 0.1184, 'compactness1': 0.2776, 'concavity1': 0.3001,
    'concave_points1': 0.1471, 'symmetry1': 0.2419, 'fractal_dimension1': 0.07871,
    'radius2': 1.095, 'texture2': 0.9053, 'perimeter2': 8.589, 'area2': 153.4,
    'smoothness2': 0.006399, 'compactness2': 0.04904, 'concavity2': 0.05373,
    'concave_points2': 0.01587, 'symmetry2': 0.03003, 'fractal_dimension2': 0.006193,
    'radius3': 25.38, 'texture3': 17.33, 'perimeter3': 184.6, 'area3': 2019.0,
    'smoothness3': 0.1622, 'compactness3': 0.6656, 'concavity3': 0.7119,
    'concave_points3': 0.2654, 'symmetry3': 0.4601, 'fractal_dimension3': 0.1189}

input_data = {}
num_columns = 3
columns = st.sidebar.columns(num_columns)

for i, feature in enumerate(feature_names):
    with columns[i % num_columns]:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(default_values.get(feature, 0.0)),
            format="%.4f",
            step=0.0001)

input_df = pd.DataFrame([input_data])

# Main Panel: Tabs

tab1, tab2 = st.tabs(["Prediction", "Input Review"])

with tab1:
    st.subheader("Step 3: Make Prediction")
    if st.button("Predict"):
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[:, 1][0] if hasattr(model, 'predict_proba') else None

        if prediction == 1:
            st.markdown("<h2 style='color:red;'>Prediction: Malignant (Cancerous)</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Prediction: Benign (Non-cancerous)</h2>", unsafe_allow_html=True)

        if prediction_proba is not None:
            st.info(f"Probability of Malignant: {prediction_proba:.4f}")
            st.info(f"Probability of Benign: {1 - prediction_proba:.4f}")

with tab2:
    st.subheader("Step 4: Review Entered Features")
    st.dataframe(input_df.style.format("{:.4f}"))
    st.caption("You can adjust the values in the sidebar to see updated predictions.")

