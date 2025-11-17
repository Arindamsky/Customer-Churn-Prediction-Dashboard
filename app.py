# Gender -> 1 Female 0 Male 
# Churn -> 1 Yes 0 No
# scaler is saved as scaler.pkl
# model is saved as model.pkl
# Order of the x -> 'Age', 'Gender', 'Tenure', 'MonthlyCharges'

import streamlit as st
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")   # fixed st.tille

st.divider()

st.write("Please enter the values and hit the Predict button")

st.divider()

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)

tenure = st.number_input("Enter Tenure (in months)", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input("Enter Monthly Charges", min_value=0.0, max_value=10000.0, value=150.0)

gender = st.selectbox("Select Gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict")   # fixed spelling

st.divider()

if predictbutton:

    gender_selected = 1 if gender == "Female" else 0

    x = [age, gender_selected, tenure, monthlycharge]

    x_array = np.array(x).reshape(1, -1)

    x_scaled = scaler.transform(x_array)

    prediction = model.predict(x_scaled)[0]

    predicted = "Yes" if prediction == 1 else "No"

    st.write(f"### 🟢 The predicted result is: **{predicted}**")

else:
    st.write("Click on Predict button to get the result")




# if predeictbutton:

#     gender_selected= 1 if gender=="Female" else 0

#     x=[age,gender_selected,tenure,monthlycharge]

#     x1= np.array(x)

#     x_array=scaler.transform([x1])

#     prediction= model.predict(x_array)[0]

#     predicted="Churn" if prediction==1 else "Not Churn"

#     st.write(f"The predicted result is : {predicted}")

# else:
#     st.write("Click on Predict button to get the result")

