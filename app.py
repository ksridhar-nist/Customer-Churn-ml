import streamlit as st
import pandas as pd
import joblib

# Load trained model (we'll save it in next step)
model = joblib.load("model.pkl")

st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details to predict churn:")

# Example input fields (you can expand later based on dataset features)
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Make a prediction
if st.button("Predict"):
    input_df = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                            columns=["tenure", "MonthlyCharges", "TotalCharges"])
    
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (probability: {prob:.2f})")
