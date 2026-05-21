import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

st.set_page_config(page_title="SmartCar Pricing Engine", layout="centered", page_icon="🚗")

st.title("🚗 SmartCar Pricing Predictor")
st.markdown("Use this interactive interface to query machine learning valuations computed against real auto marketplace records.")

# 1. CORE BRAIN: Train the model directly inside the app session memory
@st.cache_resource # Trains ONLY ONCE when the app starts
def load_and_train_model():
    # Load the raw dataset directly from your repo
    df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")
    
    # Strip any hidden whitespaces from column names just in case
    df.columns = df.columns.str.strip()
    
    # Process text sequences to isolate Brand and Model Series
    df['Brand'] = df['name'].str.split().str[0]
    df['Model'] = df['name'].str.split().str[1]
    df.dropna(subset=['Brand', 'Model', 'selling_price', 'km_driven', 'year'], inplace=True)
    
    # Process age transformations
    current_year = 2026
    df['Car_Age'] = current_year - df['year']
    
    # DEFINE EXACT FEATURE TRAINING ORDER
    feature_cols = ['km_driven', 'Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner', 'Car_Age']
    X = df[feature_cols]
    y = df['selling_price']
    
    # Create the dynamic choice dictionaries for the front-end forms
    brand_model_map = {}
    for brand in df['Brand'].unique():
        brand_model_map[brand] = sorted(df[df['Brand'] == brand]['Model'].unique().tolist())
        
    options = {
        'brand_model_map': brand_model_map,
        'fuel': sorted(df['fuel'].unique().tolist()),
        'seller_type': sorted(df['seller_type'].unique().tolist()),
        'transmission': sorted(df['transmission'].unique().tolist()),
        'owner': sorted(df['owner'].unique().tolist())
    }
    
    # Build the Sklearn Pipeline Architecture
    numeric_features = ['km_driven', 'Car_Age']
    categorical_features = ['Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    # Fit the model pipeline safely
    model_pipeline.fit(X, y)
    return model_pipeline, options

# Run the training function
with st.spinner("🚀 Training machine learning models on the cloud dataset matrix... Please hold on..."):
    try:
        model, options = load_and_train_model()
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")
        st.stop()

# 2. GENERATE FORM LAYOUT UI
st.subheader("Vehicle Specifications Form")
col1, col2 = st.columns(2)

with col1:
    brand_list = sorted(list(options['brand_model_map'].keys()))
    car_brand = st.selectbox("Manufacturer / Company", options=brand_list)
    
    available_models = options['brand_model_map'][car_brand]
    car_model = st.selectbox("Vehicle Model Line", options=available_models)
    
    manufacture_year = st.number_input("Year of Manufacture", min_value=1992, max_value=2026, value=2016, step=1)

with col2:
    km_driven = st.number_input("Total Kilometers Clocked (Odometer)", min_value=0, max_value=500000, value=55000, step=5000)
    fuel_type = st.selectbox("Fuel Engine Configuration", options=options['fuel'])
    seller_type = st.selectbox("Retail Channel Profile", options=options['seller_type'])
    transmission = st.selectbox("Gearbox Transmission Class", options=options['transmission'])
    owner_type = st.selectbox("Ownership Sequence History", options=options['owner'])

current_year = 2026
car_age = current_year - manufacture_year

# Construct user payload DataFrame
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

# CRITICAL FIX: Explicitly re-align columns to match the exact order used during model training (.fit)
feature_order = ['km_driven', 'Brand', 'Model', 'fuel', 'seller_type', 'transmission', 'owner', 'Car_Age']
input_payload = input_payload[feature_order]

st.markdown("---")

if st.button("Evaluate Fair Market Listing Price", type="primary"):
    prediction = model.predict(input_payload)[0]
    final_output = max(15000, prediction)
    st.success(f"### Predicted Market Price Valuation: ₹ {final_output:,.2f}")
    st.caption("Engineering System Node: Column feature arrays were strictly indexed, continuous fields standard-scaled, and categories hot-encoded successfully.")
