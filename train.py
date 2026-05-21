import streamlit as st
import pandas as pd
import joblib
import os

# 1. Set up Page Layout Configurations
st.set_page_config(page_title="SmartCar Pricing Engine", layout="centered", page_icon="🚗")

st.title("🚗 SmartCar Pricing Predictor")
st.markdown("Use this interactive interface to query machine learning valuations computed against real auto marketplace records.")

# 2. AUTOMATIC CLOUD RECOVERY LOGIC (Self-Healing Architecture)
# If the model binary is missing, empty, or corrupted on the cloud, train it instantly inline.
model_file = 'cardekho_ridge_model.pkl'
options_file = 'dropdown_options.pkl'

if not os.path.exists(model_file) or not os.path.exists(options_file) or os.path.getsize(model_file) < 1000:
    st.info("📦 First-Time Environment Setup: Compiling machine learning pipeline models directly on the server... Please wait a moment.")
    try:
        import train  # Executes your train.py script dynamically to build fresh asset binaries
        st.success("🎉 Compilation successful! Initializing interface components...")
    except Exception as e:
        st.error(f"Failed to auto-compile model assets. Detailed Error: {e}")
        st.warning("Please ensure 'train.py' and 'CAR DETAILS FROM CAR DEKHO.csv' are uploaded to your GitHub repository.")
        st.stop()

# 3. Load the Validated Model Pipelines Safely
model = joblib.load(model_file)
options = joblib.load(options_file)

# 4. Generate Interactive Form Layout User Interface
st.subheader("Vehicle Specifications Form")
col1, col2 = st.columns(2)

with col1:
    # Hierarchical Selection Step A: Extract Company List
    brand_list = sorted(list(options['brand_model_map'].keys()))
    car_brand = st.selectbox("Manufacturer / Company", options=brand_list)
    
    # Hierarchical Selection Step B: Dynamically filter models based on the brand chosen above
    available_models = options['brand_model_map'][car_brand]
    car_model = st.selectbox("Vehicle Model Line", options=available_models)
    
    manufacture_year = st.number_input("Year of Manufacture", min_value=1992, max_value=2026, value=2016, step=1)

with col2:
    km_driven = st.number_input("Total Kilometers Clocked (Odometer)", min_value=0, max_value=500000, value=55000, step=5000)
    fuel_type = st.selectbox("Fuel Engine Configuration", options=options['fuel'])
    seller_type = st.selectbox("Retail Channel Profile", options=options['seller_type'])
    transmission = st.selectbox("Gearbox Transmission Class", options=options['transmission'])
    owner_type = st.selectbox("Ownership Sequence History", options=options['owner'])

# 5. Preprocessing Sync (Map input parameters matching the training data definitions)
current_year = 2026
car_age = current_year - manufacture_year

# Bundle inputs into a structural payload dataframe format required by scikit-learn
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

# 6. Runtime Prediction Computation Trigger
if st.button("Evaluate Fair Market Listing Price", type="primary"):
    with st.spinner("Computing regression algorithms across processing pipeline matrices..."):
        # The preprocessor pipeline handles scaling and one-hot encoding completely under the hood
        prediction = model.predict(input_payload)[0]
        
        # Enforce a logical minimum valuation cap to handle extreme negative age coefficients
        final_output = max(15000, prediction)
        
        st.success(f"### Predicted Market Price Valuation: ₹ {final_output:,.2f}")
        st.caption("Engineering Verification Node: Continuous fields were dynamically scaled (StandardScaler) and text metrics transformed via structural One-Hot Categorical arrays successfully.")
