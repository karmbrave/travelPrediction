import streamlit as st
import pandas as pd
import joblib
import os

# Set Page Configuration
st.set_page_config(
    page_title="Travel Package Purchase Prediction",
    page_icon="✈️",
    layout="centered"
)

# App Title & Description
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.write("""
This application predicts whether a customer will purchase the newly introduced **Wellness Tourism Package**
based on their profile and interaction data.
""")

# Resolve model path
model_path = "tourism_project/deployment/best_model.pkl"
if not os.path.exists(model_path):
    # Fallback to local path if directory structure differs slightly in deployment
    model_path = "best_model.pkl"

# Load the Pre-trained Pipeline
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading the model from {path}: {e}")
        return None

model = load_model(model_path)

if model is not None:
    st.header("Customer Information Form")

    # Organize inputs into columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 18, 100, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=25000)
        city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)

    with col2:
        st.subheader("Travel & Interactions")
        type_of_contact = st.selectbox("Type of Contact", ["Self Inquiry", "Company Invited"])
        product_pitched = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
        duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=0, value=15)
        number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        number_of_person = st.slider("Number of Persons Accompanying", 1, 5, 2)
        number_of_children = st.slider("Number of Children Accompanying (Age < 5)", 0, 3, 0)
        number_of_trips = st.slider("Average Annual Trips", 1, 10, 3)
        property_star = st.selectbox("Preferred Property Star Rating", [3, 4, 5], index=0)

    st.subheader("Ownership & Credentials")
    c3, c4 = st.columns(2)
    with c3:
        passport = st.checkbox("Has a valid passport?", value=False)
    with c4:
        own_car = st.checkbox("Owns a car?", value=False)

    # Map checkbox booleans to binary integer parameters
    passport_val = 1 if passport else 0
    own_car_val = 1 if own_car else 0

    # Prepare features dictionary matching training features exactly
    input_data = {
        'Age': [age],
        'TypeofContact': [type_of_contact],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_of_pitch],
        'Occupation': [occupation],
        'Gender': [gender],
        'NumberOfPersonVisiting': [number_of_person],
        'NumberOfFollowups': [number_of_followups],
        'ProductPitched': [product_pitched],
        'PreferredPropertyStar': [property_star],
        'MaritalStatus': [marital_status],
        'NumberOfTrips': [number_of_trips],
        'Passport': [passport_val],
        'PitchSatisfactionScore': [pitch_satisfaction],
        'OwnCar': [own_car_val],
        'NumberOfChildrenVisiting': [number_of_children],
        'Designation': [designation],
        'MonthlyIncome': [monthly_income]
    }

    input_df = pd.DataFrame(input_data)

    # Add a prediction button
    st.markdown("--- ")
    if st.button("🎯 Predict Purchase Likelihood"):
        # Ensure we match the model's feature names exactly
        try:
            # Fetch the features that the model expects
            if hasattr(model, 'feature_names_in_'):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = list(model.steps[0][1].feature_names_in_)
            
            # If 'Unnamed: 0' is expected by the trained pipeline, add it dynamically as dummy
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match expectations
            input_df = input_df[expected_features]
        except Exception:
            pass
            
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.success(f"🎉 **Highly Likely to Buy!** (Confidence: {probability:.2%})")
            st.balloons()
        else:
            st.info(f"❌ **Unlikely to Buy.** (Confidence: {1 - probability:.2%})")
else:
    st.warning("⚠️ Please wait for the ML pipeline to train and save the model file.")
