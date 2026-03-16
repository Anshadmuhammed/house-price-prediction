import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("model.pkl not found. Please run  `python train.py`  first.")
        st.stop()
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
model = data["model"]

st.title("🏠 House Price Prediction")
st.markdown("Fill in the details below to get an estimated house price.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    area      = st.number_input("Area (sq ft)",     min_value=100,  max_value=10000, value=1500, step=50)
    bedrooms  = st.slider("Bedrooms",               min_value=1,    max_value=10,    value=3)
    bathrooms = st.slider("Bathrooms",              min_value=1,    max_value=6,     value=2)
    age       = st.slider("Age of Property (years)",min_value=0,    max_value=100,   value=5)

with col2:
    floors    = st.slider("Number of Floors",       min_value=1,    max_value=5,     value=1)
    garage    = st.selectbox("Garage",              ["Yes", "No"])
    garden    = st.selectbox("Garden",              ["Yes", "No"])
    location  = st.selectbox("Location",            ["Urban", "Suburban", "Rural"])
    condition = st.selectbox("Property Condition",  ["Excellent", "Good", "Fair", "Poor"])

st.markdown("---")

if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame([{
        "area_sqft":  area,
        "bedrooms":   bedrooms,
        "bathrooms":  bathrooms,
        "age_years":  age,
        "floors":     floors,
        "garage":     1 if garage == "Yes" else 0,
        "garden":     1 if garden == "Yes" else 0,
        "location":   location,
        "condition":  condition,
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"### Estimated Price: ₹ {prediction:,.0f}")
    st.info(
        f"A **{bedrooms}-bedroom, {bathrooms}-bathroom** house in a **{location}** area "
        f"with **{area} sq ft** area is estimated at **₹ {prediction:,.0f}**."
    )

st.caption("Built with Python · Scikit-learn · Streamlit")
