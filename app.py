import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="SmartCar Pricing", layout="centered")
st.title("🚗 SmartCar Pricing Predictor")
st.markdown("Use this interface to query prices computed against real CarDekho historic market metrics.")

try:
    model = joblib.load('cardekho_ridge_model.pkl')
    options = joblib.load('dropdown_options.pkl')
except FileNotFoundError:
    st.error("Missing internal parameters! Please run 'python train.py' in your terminal first.")
    st.stop()

st.subheader("Vehicle Features Form")
col1, col2 = st.columns(2)

with col1:
    # 1. User picks the Company Brand (e.g., Volkswagen)
    brand_list = sorted(list(options['brand_model_map'].keys()))
    car_brand = st.selectbox("Manufacturer / Company", options=brand_list)
    
    # 2. Dynamic Update: Pull only the models that belong to the chosen Company
    available_models = options['brand_model_map'][car_brand]
    car_model = st.selectbox("Vehicle Model Line", options=available_models)
    
    manufacture_year = st.number_input("Year of Manufacture", min_value=1992, max_value=2026, value=2016, step=1)

with col2:
    km_driven = st.number_input("Total Kilometers Clocked", min_value=0, max_value=500000, value=55000, step=5000)
    fuel_type = st.selectbox("Fuel Layout Type", options=options['fuel'])
    seller_type = st.selectbox("Retail Channel Profile", options=options['seller_type'])
    transmission = st.selectbox("Gearbox Type Class", options=options['transmission'])
    owner_type = st.selectbox("Ownership Sequence History", options=options['owner'])

current_year = 2026
car_age = current_year - manufacture_year

# Construct payload matching the training features exactly
input_payload = pd.DataFrame([{
    'km_driven': km_driven,
    'Brand': car_brand,
    'Model': car_model,
    'fuel': fuel_type,
    'seller_type': seller_type,
    'transmission': transmission,
    'owner': owner_type,
    'Car_Age': car_age
}])

st.markdown("---")

if st.button("Evaluate Fair Market Listing Price", type="primary"):
    prediction = model.predict(input_payload)[0]
    final_output = max(15000, prediction) # Cap floor price limits logically
    st.success(f"### Predicted Market Price Valuation: ₹ {final_output:,.2f}")