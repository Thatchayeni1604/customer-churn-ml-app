import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load trained model and features
# -------------------------------
model = joblib.load("notebooks/churn_model.pkl")
features = joblib.load("notebooks/features.pkl")

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("📊 Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on usage patterns.")

st.divider()

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Customer Details")

account_length = st.number_input("Account Length", min_value=0, value=100)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, value=1)

total_day_minutes = st.number_input("Total Day Minutes", value=180.0)
total_day_calls = st.number_input("Total Day Calls", value=100)
total_day_charge = st.number_input("Total Day Charge", value=30.0)

total_eve_minutes = st.number_input("Total Evening Minutes", value=200.0)
total_eve_calls = st.number_input("Total Evening Calls", value=100)
total_eve_charge = st.number_input("Total Evening Charge", value=17.0)

total_night_minutes = st.number_input("Total Night Minutes", value=200.0)
total_night_calls = st.number_input("Total Night Calls", value=100)
total_night_charge = st.number_input("Total Night Charge", value=9.0)

total_intl_minutes = st.number_input("Total Intl Minutes", value=10.0)
total_intl_calls = st.number_input("Total Intl Calls", value=3)
total_intl_charge = st.number_input("Total Intl Charge", value=2.7)

international_plan = st.selectbox("International Plan", ["no", "yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["no", "yes"])

st.divider()

# -------------------------------------------------
# Build INPUT DATAFRAME (MATCHES TRAINING FEATURES)
# -------------------------------------------------
# Create empty dataframe with ALL training features
input_df = pd.DataFrame(0, index=[0], columns=features)

# Fill numerical features
input_df["account_length"] = account_length
input_df["number_customer_service_calls"] = customer_service_calls

input_df["total_day_minutes"] = total_day_minutes
input_df["total_day_calls"] = total_day_calls
input_df["total_day_charge"] = total_day_charge

input_df["total_eve_minutes"] = total_eve_minutes
input_df["total_eve_calls"] = total_eve_calls
input_df["total_eve_charge"] = total_eve_charge

input_df["total_night_minutes"] = total_night_minutes
input_df["total_night_calls"] = total_night_calls
input_df["total_night_charge"] = total_night_charge

input_df["total_intl_minutes"] = total_intl_minutes
input_df["total_intl_calls"] = total_intl_calls
input_df["total_intl_charge"] = total_intl_charge

# Handle ONE-HOT encoded categorical features
if international_plan == "yes" and "international_plan_yes" in input_df.columns:
    input_df["international_plan_yes"] = 1

if voice_mail_plan == "yes" and "voice_mail_plan_yes" in input_df.columns:
    input_df["voice_mail_plan_yes"] = 1

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn\n\n**Probability:** {probability:.2%}")
    else:
        st.success(f"✅ Customer is likely to stay\n\n**Churn Probability:** {probability:.2%}")
