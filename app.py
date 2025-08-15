import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown  # <-- Using gdown for reliable downloads
import os

# Sklearn imports are needed for the custom classes to work
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# IMPORTANT: The custom classes and functions used in the pipeline
# must be defined in the script where you load the pipeline.
# ==============================================================================

# Helper function that was used in the training pipeline
def clip_at_zero(x):
    """Replaces all negative values in a numpy array with 0."""
    return np.maximum(0, x)

# Custom Transformer for initial data cleaning
class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def _remove_underscore(self, value):
        s = str(value).strip()
        if s.startswith('_'):
            s = s.lstrip('_')
        if s.endswith('_'):
            s = s.rstrip('_')
        return np.nan if s in ['', 'nan'] else float(s)
        
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(self._remove_underscore)
        return X_copy

# Custom Transformer for 'Type_of_Loan' feature engineering
class LoanTypeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.loan_types = [
            'Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 
            'Home Equity Loan', 'Mortgage Loan', 'Student Loan', 
            'Debt Consolidation Loan', 'Payday Loan'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.to_frame()
        col_name = X_copy.columns[0]
        loan_data = X_copy[col_name].fillna('')
        for loan in self.loan_types:
            loan_key = loan.split(' ')[0].replace('-', '_')
            X_copy[loan_key] = loan_data.str.contains(loan.split(' ')[0], case=False, na=False).astype(int)
        return X_copy.drop(columns=[col_name]).values

# ==============================================================================
# Function to Load Model using gdown (Most Reliable Method)
# ==============================================================================

@st.cache_resource
def load_model():
    """Download the model from Google Drive using gdown and load it."""
    model_path = 'stacking_classifier_pipeline.pkl'
    
    if not os.path.exists(model_path):
        file_id = "15xF60zDjFdGGvPcFb9EqLSepXKAIX73A"
        
        try:
            with st.spinner("Downloading model... (this may take a minute on first run)"):
                # Use gdown to handle the download reliably
                gdown.download(id=file_id, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model using gdown: {e}")
            st.error("Please ensure the Google Drive file is shared with 'Anyone with the link'.")
            return None

    # Load the pipeline from the local file
    try:
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)
        return pipeline
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        # If loading fails, remove the potentially corrupt file to force a re-download next time
        if os.path.exists(model_path):
            os.remove(model_path)
        return None

# Load the model
pipeline = load_model()

# ==============================================================================
# Streamlit User Interface
# ==============================================================================

st.set_page_config(page_title="Credit Score Prediction", layout="wide")
st.title("Credit Score Prediction App 💳")
st.markdown("Enter the customer details below to predict their credit score. The model will classify the score as **Poor, Standard, or Good**.")

# Create columns for layout
col1, col2, col3 = st.columns(3)

# --- Input Fields ---
with col1:
    st.subheader("Personal Information")
    age = st.text_input("Age", "35")
    occupation = st.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect'])
    month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August'])

with col2:
    st.subheader("Financial Details")
    annual_income = st.text_input("Annual Income (e.g., _150000_)", "_150000_")
    monthly_inhand_salary = st.number_input("Monthly In-hand Salary", value=12500.0, format="%.2f")
    outstanding_debt = st.text_input("Outstanding Debt (e.g., _800_)", "_800_")
    monthly_balance = st.number_input("Monthly Balance", value=1500.0, format="%.2f")
    
with col3:
    st.subheader("Credit & Loan Information")
    num_of_loan = st.text_input("Number of Loans", "2")
    type_of_loan = st.text_input("Type of Loans (comma-separated)", "Personal Loan, Auto Loan")
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", value=2.0, format="%.1f")
    num_of_delayed_payment = st.text_input("Number of Delayed Payments", "4")


st.divider()

# --- More Input Fields in a second row ---
col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("Account Details")
    num_bank_accounts = st.number_input("Number of Bank Accounts", value=3, step=1)
    num_credit_card = st.number_input("Number of Credit Cards", value=4, step=1)
    
with col5:
    st.subheader("Investment & Payments")
    amount_invested_monthly = st.text_input("Amount Invested Monthly (e.g., _200.50_)", "_200.50_")
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["Yes", "No", "NM"])
    total_emi_per_month = st.number_input("Total EMI per Month", value=350.0, format="%.2f")

with col6:
    st.subheader("Behavior & History")
    interest_rate = st.number_input("Interest Rate (%)", value=6, step=1)
    changed_credit_limit = st.text_input("Changed Credit Limit", "5000.0")
    #st.text_input("Credit Mix (Not Used by Model)", "Good", disabled=True)
    # st.text_input("Payment Behaviour (Not Used by Model)", "High_spent_Small_value_payments", disabled=True)


# --- Prediction Button ---
if st.button("Predict Credit Score", type="primary", use_container_width=True):
    if pipeline is not None:
        input_data = pd.DataFrame({
            'Age': [age], 'Occupation': [occupation], 'Annual_Income': [annual_income],
            'Monthly_Inhand_Salary': [monthly_inhand_salary], 'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_card], 'Interest_Rate': [interest_rate],
            'Num_of_Loan': [num_of_loan], 'Type_of_Loan': [type_of_loan],
            'Num_Credit_Inquiries': [num_credit_inquiries], 'Credit_Mix': ['Good'],
            'Outstanding_Debt': [outstanding_debt], 'Credit_Utilization_Ratio': [0.35],
            'Credit_History_Age': ['20 Years and 5 Months'], 'Payment_of_Min_Amount': [payment_of_min_amount],
            'Total_EMI_per_month': [total_emi_per_month], 'Amount_invested_monthly': [amount_invested_monthly],
            'Payment_Behaviour': ['High_spent_Small_value_payments'], 'Num_of_Delayed_Payment': [num_of_delayed_payment],
            'Changed_Credit_Limit': [changed_credit_limit], 'Monthly_Balance': [monthly_balance],
            'Month': [month], 'ID': ['CUS_001'], 'Customer_ID': ['ACC_001'],
            'Name': ['John Doe'], 'SSN': ['123-45-678']
        })
        
        try:
            prediction_encoded = pipeline.predict(input_data)
            prediction_proba = pipeline.predict_proba(input_data)
            
            reverse_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
            predicted_label = reverse_mapping[prediction_encoded[0]]
            
            st.subheader("Prediction Result")
            if predicted_label == 'Good':
                st.success(f"The predicted credit score is: **Good** 👍")
            elif predicted_label == 'Standard':
                st.info(f"The predicted credit score is: **Standard** 👌")
            else:
                st.error(f"The predicted credit score is: **Poor** 👎")

            st.write("Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Credit Score': ['Poor', 'Standard', 'Good'],
                'Probability': prediction_proba[0]
            })
            st.dataframe(prob_df.style.format({'Probability': '{:.2%}'}), use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model could not be loaded. Please check the error messages above.")

