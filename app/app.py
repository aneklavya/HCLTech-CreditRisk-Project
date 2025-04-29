import streamlit as st
import joblib
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import os

# Set page configuration
st.set_page_config(page_title="Loan Predictor", page_icon="💡", layout="centered")

# Function to fetch animations
def load_lottieurl(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load animations
approved_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_u4yrau.json")  # Approval animation
denied_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_edo8zvm7.json")  # Denial animation

# Load the trained model

model_path = os.path.join('models', 'best_model.pkl')
model = joblib.load(model_path)

# Define the prediction function
def predict(loan_amount, annual_income, loan_to_value, debt_ratio, credit_score):
    inputs = np.array([loan_amount, annual_income, loan_to_value, debt_ratio, credit_score]).reshape(1, -1)
    inputs_df = pd.DataFrame(inputs, columns=["loan_amount", "annual_income", "loan_to_value", "debt_ratio", "credit_score"])
    prediction = model.predict(inputs_df)
    return "Denied" if prediction[0] == 1 else "Approved"

# Calculate risk level
def calculate_risk_level(loan_amount, annual_income, loan_to_value, debt_ratio, credit_score):
    loan_weight = 0.3
    income_weight = 0.2
    ltv_weight = 0.2
    debt_weight = 0.2
    credit_weight = 0.1

    risk_score = (
        (loan_amount / 100000) * loan_weight +
        (1 - annual_income / 1000000) * income_weight +
        loan_to_value * ltv_weight +
        debt_ratio * debt_weight +
        (1 - credit_score / 850) * credit_weight
    )
    return risk_score * 100

# Apply custom CSS for Copilot-inspired dark mode
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background-color: #1E1E1E;
            color: #E0E0E0;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #72B3D0;
            margin-bottom: 1rem;
        }
        .description {
            font-size: 1.2rem;
            color: #AAAAAA;
            margin-bottom: 1.5rem;
        }
        .risk-level {
            font-size: 1.5rem;
            color: #FF9800;
            margin: 1rem 0;
        }
        .stButton>button {
            background-color: #72B3D0;
            color: white;
            border-radius: 10px;
            padding: 10px 25px;
            font-size: 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #63A6C1;
            transform: scale(1.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App layout
st.markdown(f'<h1 class="main-title">🌟 Loan Default Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="description">Fill in the details below to get a prediction on whether the loan will be <strong>Approved</strong> or <strong>Denied</strong>, and see the associated risk level.</p>', unsafe_allow_html=True)

# Input sliders
loan_amount = st.slider("💵 Loan Amount", min_value=0, max_value=100000, step=100, value=33000)
annual_income = st.slider("💰 Annual Income", min_value=0, max_value=1000000, step=1000, value=50000)
loan_to_value = st.slider("📊 Loan-to-Value Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.7)
debt_ratio = st.slider("📉 Debt Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.4)
credit_score = st.slider("📈 Credit Score", min_value=0, max_value=850, step=1, value=650)

# Button for prediction
if st.button("💡 Predict"):
    # Predict loan approval
    result = predict(loan_amount, annual_income, loan_to_value, debt_ratio, credit_score)

    # Calculate risk level
    risk_level = calculate_risk_level(loan_amount, annual_income, loan_to_value, debt_ratio, credit_score)

    # Display prediction result
    if result == "Approved":
        st.success(f"✅ **Prediction: {result}** – Congratulations! Your loan is likely to be approved.")
        if approved_lottie:
            st_lottie(approved_lottie, height=300)
    else:
        st.error(f"❌ **Prediction: {result}** – Sorry, your loan is likely to be denied.")
        if denied_lottie:
            st_lottie(denied_lottie, height=300)
