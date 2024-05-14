import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the data
salary_data = pd.read_csv(r"C:\Users\91822\OneDrive\Desktop\XX\Salary_Data.csv")  # Update with the path to your data file

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(salary_data[['YearsExperience']])
y = salary_data['Salary']

# Train the model
model = LinearRegression()
model.fit(X_scaled, y)

# Streamlit app
st.title("Salary Prediction App")

# Centered input for years of experience
st.markdown("<h3 style='color: purple; font-size: 20px;'>Input Years of Experience</h3>", unsafe_allow_html=True)

years_exp_input = st.text_input("Years of Experience", "Enter years of experience")

# Predict the salary
if st.button("Predict Salary"):
    try:
        years_exp_input = float(years_exp_input)
        scaled_input = scaler.transform([[years_exp_input]])
        predicted_salary = model.predict(scaled_input)
        formatted_salary = "${:,.0f}".format(predicted_salary[0])
        st.markdown(f"<p style='color:green; font-size:30px'><b>Predicted Salary:</b> {formatted_salary}</p>", unsafe_allow_html=True)
    except ValueError:
        st.error("Please enter a valid number for years of experience.")