# === IMPORTS ===
import streamlit as st
import pandas as pd
import joblib

# Set Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🩺")

# === HEADER ===
st.title("🩺 Breast Cancer Detection App")
st.write("Select a row from the dataset to predict breast cancer diagnosis.")

# === LOAD MODEL AND SCALER ===
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("breast_cancer_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# === LOAD DATASET ===
@st.cache_data
def load_data():
    df = pd.read_csv("Cancer_Data.csv")
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')  # Drop unnecessary columns
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Convert target to numeric
    return df

df = load_data()

# === DISPLAY DATASET ===
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10))

# === SELECT ROW FOR PREDICTION ===
st.subheader("🔍 Select a Row to Predict")
row_index = st.number_input("Enter Row Index (0 to {}):".format(len(df)-1), min_value=0, max_value=len(df)-1, value=0, step=1)

# === MAKE PREDICTION ===
if st.button("🎯 Predict Diagnosis"):
    feature_columns = df.columns.drop('diagnosis')
    selected_row = df.iloc[[row_index]][feature_columns]  # DataFrame, not Series
    scaled_input = scaler.transform(selected_row)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    diagnosis = "Malignant" if prediction == 1 else "Benign"
    confidence = probability if prediction == 1 else (1 - probability)

    # === DISPLAY RESULT ===
    st.subheader("🧾 Diagnosis Result")
    st.write(f"**Diagnosis:** {diagnosis}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    if diagnosis == "Malignant":
        st.error("⚠️ High Risk: Malignant Tumor Detected.")
    else:
        st.success("✅ Benign Tumor: Low Risk.")

# === FOOTER ===
st.markdown("---")
st.caption("Using Random Forest model trained on Breast Cancer Wisconsin dataset.")
