import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000/predict")

# ==========================================================================
# Page Configuration
# ==========================================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Sidebar Configuration
# ==========================================================================
with st.sidebar:
    st.title("🛡️Fraud Detection System")

    st.caption("Detect fraudulent financial transactions using a trained Machine Learning model.")

    st.divider()


# ==========================================================================
# Tabs Configuration
# ==========================================================================
predict_tab, documentation_tab, about_tab = st.tabs(["Prediction", "Documentation", "About"])


# ==========================================================================
# Prediction Tab
# ==========================================================================
with predict_tab:

    st.header("Prediction")

    st.write("Enter the transaction details below.")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            step = st.number_input("Step", min_value=1, max_value=6, value=1)
            amount = st.number_input("Amount", min_value=0.0, value=0.0)
            oldbalanceOrg = st.number_input("Old Balance Origin", min_value=0.0, value=0.0)
            newbalanceDest = st.number_input("New Balance Destination", min_value=0.0, value=0.0)
            nameDest = st.text_input("Destination Account ID", value="")

        with col2:
            transaction_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
            oldbalanceDest = st.number_input("Old Balance Destination", min_value=0.0, value=0.0)
            newbalanceOrig = st.number_input("New Balance Origin", min_value=0.0, value=0.0)
            nameOrig = st.text_input("Origin Account ID", value="")
            submitted = st.form_submit_button("Predict", use_container_width=True)
        if submitted:
            data = {
                "step": step,
                "type": transaction_type,
                "amount": amount,
                "nameOrig": nameOrig,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "nameDest": nameDest,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest
            }

            try:
                response = requests.post(API_URL, json=data)
                response.raise_for_status()
                result = response.json()
                prediction = result["predictions"]
                prediction_class = result["class"]
                status = result["status"]
                st.divider()
                st.subheader("Prediction Result")
                if prediction == 'Fraud':
                    st.error(f"🚨 Prediction: {prediction}")
                else:
                    st.success(f"✅ Prediction: {prediction}")

                col1, col2 = st.columns(2)
                
                with col2:
                    st.success(f"Request Status: {status.capitalize()}")

            except requests.exceptions.HTTPError:
                st.error(f"HTTP Error: {response.status_code}")
                st.json(response.json())

            except Exception as e:
                st.error(f"Unexpected Error: {e}")




# ==========================================================
# Documentation Tab
# ==========================================================
with documentation_tab:

    st.header("Documentation")

    st.markdown("""
## Project Overview

This application predicts whether a financial transaction is likely to be fraudulent using a trained XGBoost machine learning model.

The application accepts transaction details, sends them to a FastAPI backend, and returns a fraud prediction in real time.
""")

    st.markdown("""
### Supported Transaction Types

- CASH_IN
- CASH_OUT
- DEBIT
- PAYMENT
- TRANSFER
""")

# ==========================================================
# About Tab
# ==========================================================
with about_tab:

    st.header("About")

    st.header("Project Overview")

    st.markdown("""
This application uses a trained **XGBoost Machine Learning model** to detect fraudulent
financial transactions.

The user enters transaction details through the web interface.
The application sends the information to a FastAPI backend, where the trained model
evaluates the transaction and returns a prediction in real time.

The prediction indicates whether the transaction is likely to be:

- ✅ Legitimate Transaction
- 🚨 Fraudulent Transaction
""")

    st.divider()

    st.subheader("Technology Stack")

    st.markdown("""
- **Frontend:** Streamlit
- **Backend API:** FastAPI
- **Machine Learning Model:** XGBoost
- **Model Serialization:** Joblib
- **Experiment Tracking:** MLflow
- **Programming Language:** Python
""")

    st.divider()

    st.subheader("Input Features")

    st.markdown("""
The model expects the following transaction information:

- Step
- Transaction Type
- Amount
- Origin Account ID
- Origin Old Balance
- Origin New Balance
- Destination Account ID
- Destination Old Balance
- Destination New Balance
""")
