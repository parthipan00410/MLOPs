import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Correct HuggingFace model download
model_path = hf_hub_download(
    repo_id="Parthipan00410/churn-model",
    filename="best_churn_model.joblib"
)

# Load model
model = joblib.load(model_path)

# UI
st.title("Customer Churn Prediction App")
st.write("Predict if a bank customer is likely to churn based on their profile.")

# Inputs
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=20, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=10000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])
IsActiveMember = st.selectbox("Active Member?", ["Yes", "No"])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Convert inputs
input_data = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': 1 if HasCrCard == "Yes" else 0,
    'IsActiveMember': 1 if IsActiveMember == "Yes" else 0,
    'EstimatedSalary': EstimatedSalary
}])

# Prediction
threshold = 0.45

if st.button("Predict"):
    proba = model.predict_proba(input_data)[0, 1]
    result = "Churn" if proba >= threshold else "Not Churn"
    st.subheader(f"Prediction: **{result}**")
    st.write(f"Probability: {proba:.2f}")

