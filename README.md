# Credit Card Fraud Detection

This project involves building a real-time credit card fraud detection system using machine learning and deploying it via a Streamlit app. The system uses the XGBoost classifier for prediction and SHAP (SHapley Additive exPlanations) for model interpretability.

## Project Overview

Credit card fraud detection is an important application of machine learning, especially for preventing financial fraud. In this project, we use a dataset of credit card transactions to classify them as either legitimate or fraudulent based on various features such as transaction amount, time, and more.

This project consists of two main parts:
1. **Training the Fraud Detection Model** using XGBoost.
2. **Deploying a Streamlit Web Application** for real-time fraud detection and displaying SHAP value explanations.

## Dataset

The dataset used in this project is from the [Credit Card Fraud Detection Kaggle competition](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by credit card holders and is labeled to indicate whether a transaction is fraudulent (`Class = 1`) or legitimate (`Class = 0`).

### Columns in the dataset:
- `V1` to `V28`: Features resulting from a PCA transformation.
- `Time`: The time elapsed between this transaction and the first transaction in the dataset.
- `Amount`: The transaction amount.
- `Class`: The target variable where 1 indicates fraudulent and 0 indicates legitimate transactions.

### Steps in the Project:

1. **Data Preprocessing:**
   - The dataset is cleaned and preprocessed by scaling the `Amount` feature using `StandardScaler` and handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
   
2. **Model Training:**
   - The XGBoost classifier is used to train the model on the preprocessed dataset.

3. **Model Evaluation:**
   - The model is evaluated using classification metrics such as accuracy, precision, recall, and ROC AUC score.

4. **Model Interpretability:**
   - SHAP (SHapley Additive exPlanations) is used to explain the model’s predictions, allowing us to understand which features contributed most to the model’s decision-making process.

5. **Streamlit App Deployment:**
   - A Streamlit web application is created to allow users to upload their own transaction data for real-time fraud detection. The app uses the trained model to classify transactions and displays the results along with visualizations.

## Requirements

- Python 3.x
- `kagglehub`
- `pandas`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`
- `shap`
- `matplotlib`
- `joblib`
- `streamlit`
- `plotly`

You can install the required packages using the following command:

```bash
pip install kagglehub pandas scikit-learn imbalanced-learn xgboost shap matplotlib joblib streamlit plotly


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

