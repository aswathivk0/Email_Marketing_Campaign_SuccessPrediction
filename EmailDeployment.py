#!/usr/bin/env python
# coding: utf-8

# In[27]:


import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Define features and default values
numerical_features = [
    'Customer_Age',
    'Emails_Opened',
    'Emails_Clicked',
    'Purchase_History',
    'Time_Spent_On_Website',
    'Days_Since_Last_Open',
    'Customer_Engagement_Score',
    'Clicked_Previous_Emails',
    'Device_Type'
]
default_values = [50,3,1,2443.8,5.9,25,61.8,1,0]

# Streamlit App
st.title("Email Open Rate Predictor")

# User inputs for selected features
customer_age = st.number_input("Customer Age (in years)", min_value=15, max_value=90, value=50)
emails_opened = st.number_input("Number of Emails Opened Previously", min_value=0, max_value=50, value=3)
emails_clicked = st.number_input("Number of Emails Clicked Previously", min_value=0, max_value=20, value=1)
purchase_history = st.number_input("Total Purchase History ($)", min_value=0.0, max_value=5000.0, value=2443.8)
time_spent = st.number_input("Time Spent on Website (minutes)", min_value=0.0, max_value=60.0, value=5.9)
days_since_last_open = st.number_input("Days Since Last Email Open", min_value=0, max_value=365, value=25)
engagement_score = st.number_input("Customer Engagement Score", min_value=0.0, max_value=100.0, value=61.8)
clicked_previous_emails = st.selectbox("Clicked Previous Emails?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
device_type = st.selectbox("Device Type", options=[0, 1], format_func=lambda x: "Mobile" if x == 1 else "Desktop")

# Create a dictionary of inputs
user_input = {
    'Customer_Age': customer_age,
    'Emails_Opened': emails_opened,
    'Emails_Clicked': emails_clicked,
    'Purchase_History': purchase_history,
    'Time_Spent_On_Website': time_spent,
    'Days_Since_Last_Open': days_since_last_open,
    'Customer_Engagement_Score': engagement_score,
    'Clicked_Previous_Emails': clicked_previous_emails,
    'Device_Type': device_type
}

# Fill missing features with defaults
complete_input = [user_input.get(feature, default)
                  for feature, default in zip(numerical_features, default_values)]

# Standardize inputs
input_values = np.array(complete_input).reshape(1, -1)
standardized_input = scaler.transform(input_values)

# Predict and display results
if st.button("Predict"):
    prediction = model.predict(standardized_input)
    st.write("Prediction:", "Opened" if prediction[0] == 1 else "Not Opened")
    probabilities = model.predict_proba(standardized_input)[0]  # Get probabilities
    st.write(f"Probability of 'Not Opened': {probabilities[0]:.2f}")
    st.write(f"Probability of 'Opened': {probabilities[1]:.2f}")


# In[ ]:




