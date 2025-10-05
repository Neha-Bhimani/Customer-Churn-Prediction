import streamlit as st
import numpy as np
import pandas as pd
import joblib


# Load the model
best_model = joblib.load(open("best_model.pkl", "rb"))
encoder = joblib.load(open("encoders.pkl", "rb"))


st.title("Chrun Prediction App")

st.markdown("Predict whether a customer will Churn or Not Churn based on their service details.")

st.divider()

st.subheader("Enter Customer Information:")

st.divider()

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox("Is the customer a Senior Citizen?", [0, 1])

Partner = st.selectbox("Does the customer have a Partner?", ["Yes", "No"])

Dependents = st.selectbox("Does the customer have Dependents?", ["Yes", "No"])

tenure = st.number_input("Enter the Tenure (months)", min_value=0, max_value=100, value=1)

PhoneService = st.selectbox("Does the customer have Phone Service?", ["Yes", "No"])

MultipleLines = st.selectbox("Does the customer have Multiple Lines?", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("Choose Internet Service type", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.selectbox("Does the customer have Online Security?", ["Yes", "No", "No internet service"])

OnlineBackup = st.selectbox("Does the customer have Online Backup?", ["Yes", "No", "No internet service"])

DeviceProtection = st.selectbox("Does the customer have Device Protection?", ["Yes", "No", "No internet service"])

TechSupport = st.selectbox("Does the customer have Tech Support?", ["Yes", "No", "No internet service"])

StreamingTV = st.selectbox("Does the customer use Streaming TV?", ["Yes", "No", "No internet service"])

StreamingMovies = st.selectbox("Does the customer use Streaming Movies?", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Select the Contract type", ["Month-to-month", "One year", "Two year"])

PaperlessBilling = st.selectbox("Is Paperless Billing enabled?", ["Yes", "No"])

PaymentMethod = st.selectbox("Select the Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

MonthlyCharges = st.number_input("Enter the Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)

TotalCharges = st.number_input("Enter the Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

st.divider()

input_dict = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_dict])



# -----------------------------
# Encode categorical columns
# -----------------------------
for col in input_df.columns:
    if col in encoder:
        try:
            # If encoder[col] is a sklearn LabelEncoder or similar
            input_df[col] = encoder[col].transform(input_df[col])
        except Exception:
            # If encoder[col] is a mapping dict
            input_df[col] = input_df[col].map(encoder[col]).fillna(0)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Churn"):
    pred = best_model.predict(input_df)[0]
    prob = best_model.predict_proba(input_df)[0][1] if hasattr(best_model, "predict_proba") else None

    if pred == 1:
        st.error(f"Customer is likely to **Churn**. Probability: {prob:.2f}" if prob else "Customer is likely to **Churn**.")
    else:
        st.success(f"Customer is likely to **Not Churn**. Probability: {prob:.2f}" if prob else "Customer is likely to **Not Churn**.")