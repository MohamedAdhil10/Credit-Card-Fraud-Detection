import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

# Load model and scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Real-Time Credit Card Fraud Detection System")

uploaded_file = st.file_uploader("Upload a transaction file (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load and preprocess uploaded data
    data = pd.read_csv(uploaded_file)

    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]

    data['Fraud Probability'] = probability
    data['Is Fraudulent'] = np.where(prediction == 1, 'Yes', 'No')

    st.subheader("Full Prediction Results")
    st.dataframe(data)

    st.subheader("Flagged Risky Transactions")
    risky = data[data['Is Fraudulent'] == 'Yes']
    st.dataframe(risky)

    if not risky.empty:
        st.subheader("SHAP Explanation and Risk Level for Top Risky Transaction")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data.drop(['Fraud Probability', 'Is Fraudulent'], axis=1))

        # Get Top Risky Transaction
        top_transaction = data.drop(['Fraud Probability', 'Is Fraudulent'], axis=1).iloc[0]
        top_probability = data.iloc[0]['Fraud Probability']

        # Risk Level Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=top_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': top_probability * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # SHAP Force Plot
        fig, ax = plt.subplots(figsize=(12, 4))  # Create figure
        shap.force_plot(explainer.expected_value, shap_values[0, :], top_transaction, matplotlib=True)
        st.pyplot(fig)  # Display figure properly
    else:
        st.info("No fraudulent transactions detected!")
else:
    st.info("Please upload a transaction CSV file to proceed.")
