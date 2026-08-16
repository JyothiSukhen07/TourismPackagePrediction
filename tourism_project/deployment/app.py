import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a Tourism Package.
Enter the customer details below to get a prediction.
""")

# Input fields based on the tourism.csv dataset
age = st.number_input("Age", 18, 61, 37)
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.number_input("City Tier", 1, 3, 1)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 5, 127, 15)
occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business', 'Government Sector'])
gender = st.selectbox("Gender", ['Female', 'Male'])
number_of_person_visiting = st.number_input("Number of Persons Visiting", 1, 5, 3)
number_of_followups = st.number_input("Number of Follow-ups", 1, 6, 4)
product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'Executive', 'Luxury'])
preferred_property_star = st.number_input("Preferred Property Star", 3, 5, 3)
marital_status = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married'])
number_of_trips = st.number_input("Number of Trips Annually", 1, 22, 3)
passport = st.selectbox("Has Passport? (0=No, 1=Yes)", [0, 1])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", 1, 5, 3)
own_car = st.selectbox("Owns Car? (0=No, 1=Yes)", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 3, 1)
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
monthly_income = st.number_input("Monthly Income", 1000.0, 98678.0, 23178.0, 100.0)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    result = "Customer Will Purchase" if prediction == 1 else "Customer Will Not Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
