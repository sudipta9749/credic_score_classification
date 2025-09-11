import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import gdown

warnings.filterwarnings('ignore')

# ==============================================================================
#                      LOAD THE PRE-TRAINED MODEL (from Google Drive)
# ==============================================================================
@st.cache_resource
def load_model():
    """
    Downloads and loads the stacking model from Google Drive.
    Replace the file_id with your actual Google Drive file ID.
    """
    try:
        # 🔹 Replace with your actual Google Drive file ID
        file_id = "1d6kvZpZjJwr1Wfm62sJ06T6ef7p1bZWn"
        url = f"https://drive.google.com/uc?id={file_id}"
        output_path = "final_stacking_model.pkl"

        # Download from Google Drive
        gdown.download(url, output_path, quiet=False)

        # Load model from pickle
        with open(output_path, "rb") as file:
            model_data = pickle.load(file)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the dictionary containing the model and other objects.
model_data = load_model()

if isinstance(model_data, dict):
    model = model_data['model']
    feature_columns = model_data['feature_columns']
else:
    model = None 
    st.error("The model file is not in the correct format. Please re-run the training script.")

# ==============================================================================
#                            STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="Credit Score Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("💳 Credit Score Prediction App")
st.write("""
This app uses a pre-trained **Stacking Classifier** model to predict a customer's credit score.  
**Note**: This interface expects **pre-transformed** numerical values, not raw data.
""")

# --- Sidebar for User Input ---
st.sidebar.header("Enter Customer Data (Transformed)")

if model:
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
    occupations = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
    pmt_min_amount = ['Yes', 'No', 'NM']
    
    input_data = {}
    # --- FIXED INPUTS: Changed to generic number_input for transformed values ---
    st.sidebar.markdown("### Numerical Features (Pre-Transformed)")
    input_data['Age'] = st.sidebar.number_input("Age (Transformed Value)", value=0.0, format="%.6f")
    input_data['Monthly_Inhand_Salary'] = st.sidebar.number_input("Monthly Inhand Salary (Transformed)", value=0.0, format="%.6f")
    input_data['Num_Credit_Inquiries'] = st.sidebar.number_input("Num Credit Inquiries (Transformed)", value=0.0, format="%.6f")
    input_data['Interest_Rate'] = st.sidebar.number_input("Interest Rate (Transformed)", value=0.0, format="%.6f")
    input_data['Delay_from_due_date'] = st.sidebar.number_input("Delay from due date", value=0.0)
    input_data['Num_of_Delayed_Payment'] = st.sidebar.number_input("Num of Delayed Payment (Transformed)", value=0.0, format="%.6f")
    input_data['Changed_Credit_Limit'] = st.sidebar.number_input("Changed Credit Limit", value=0.0, format="%.2f")
    input_data['Outstanding_Debt'] = st.sidebar.number_input("Outstanding Debt", value=0.0, format="%.2f")
    input_data['Total_EMI_per_month'] = st.sidebar.number_input("Total EMI per month (Transformed)", value=0.0, format="%.6f")
    input_data['Amount_invested_monthly'] = st.sidebar.number_input("Amount invested monthly (Transformed)", value=0.0, format="%.6f")
    
    st.sidebar.markdown("### Categorical Features")
    selected_month = st.sidebar.selectbox("Month", months)
    selected_occupation = st.sidebar.selectbox("Occupation", occupations)
    selected_pmt_min = st.sidebar.selectbox("Payment of Minimum Amount", pmt_min_amount)

    if st.sidebar.button("Predict Credit Score", type="primary"):
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical variables
        for month in months[1:]:
            input_df[f'Month_{month}'] = 1 if selected_month == month else 0
        for occ in occupations[1:]:
            input_df[f'Occupation_{occ}'] = 1 if selected_occupation == occ else 0
        for pmt in pmt_min_amount[1:]:
            input_df[f'Payment_of_Min_Amount_{pmt}'] = 1 if selected_pmt_min == pmt else 0
        
        # Align columns with training feature set
        final_input_df = pd.DataFrame(columns=feature_columns).fillna(0)
        final_input_df = pd.concat([final_input_df, input_df], ignore_index=True).fillna(0)
        final_input_df = final_input_df[feature_columns]

        # Make prediction
        prediction = model.predict(final_input_df)
        prediction_proba = model.predict_proba(final_input_df)

        st.subheader("Prediction Result")
        score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        predicted_score = score_mapping[prediction[0]]

        if predicted_score == 'Good':
            st.success(f"Predicted Credit Score: **{predicted_score}** ✅")
        elif predicted_score == 'Standard':
            st.warning(f"Predicted Credit Score: **{predicted_score}** 😐")
        else:
            st.error(f"Predicted Credit Score: **{predicted_score}** ❌")

        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame({
            'Credit Score': ['Poor', 'Standard', 'Good'],
            'Probability': prediction_proba[0]
        })
        st.bar_chart(proba_df.set_index('Credit Score'))
