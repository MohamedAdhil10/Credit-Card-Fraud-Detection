# Credit Card Fraud Detection System

Project Overview

This project focuses on building a real-time Credit Card Fraud Detection system using machine learning (XGBoost) and visual explanation techniques (SHAP). It includes:

Training a fraud detection model

Evaluating model performance

Developing an interactive web application with Streamlit for real-time predictions


Features

SMOTE: Handling imbalanced datasets

XGBoost Classifier: Robust and accurate model

SHAP: Model interpretability and transaction risk explanation

Streamlit App: Upload transaction data and detect frauds

Risk Gauge: Visual risk level for flagged transactions


Installation

1. Clone the repository:



https://github.com/your-repo/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install dependencies:



pip install -r requirements.txt

requirements.txt

streamlit
pandas
numpy
scikit-learn
imblearn
xgboost
matplotlib
shap
kagglehub
plotly
joblib

Dataset

The dataset used is the Credit Card Fraud Detection Dataset available on Kaggle.

How to Run

1. Train the Model

Run the training script to:

Download the dataset

Preprocess the data

Train the XGBoost model

Save the trained model and scaler


python train_model.py

2. Launch Streamlit App

Run the app locally:

streamlit run app.py

3. Upload Transaction File

Upload a CSV file containing transaction data.

The app predicts whether transactions are fraudulent.

Visual explanations for flagged transactions are generated.


File Structure

credit-card-fraud-detection/
├── train_model.py       # Training Script
├── app.py               # Streamlit Web Application
├── xgb_fraud_model.pkl  # Trained Model
├── scaler.pkl           # Scaler for 'Amount' Feature
├── requirements.txt     # Python Dependencies
├── README.md            # Project Documentation

Output

Classification Report: Precision, Recall, F1-score

ROC AUC Score: Model evaluation metric

Streamlit UI: Upload, Predict, Explain transactions


Screenshots

Fraud Risk Gauge

Shows how risky a transaction is.

SHAP Force Plot

Explains which features contributed to a transaction being flagged as fraud.

License

This project is open-source and free to use.


---

Developed with passion for solving real-world problems!

