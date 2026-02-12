import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie

import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="ProHouse Valuator", page_icon="🏢", layout="wide")

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('house_model.pkl')

model = load_model()

# --- 2. Sidebar Setup ---
with st.sidebar:
    st.title("🏢 ProHouse Valuator")
    st.markdown("---")
    st.header("App Navigation")
    app_mode = st.radio("Go to", ["Dashboard", "Live Prediction"])
    st.markdown("---")
    st.info("Built with Streamlit & Scikit-Learn")

# --- 3. Lottie Animation Function ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Lottie animation for sidebar
lottie_url = "https://lottie.host/cbe12934-49ff-4200-8a28-98bcb8ce30e5/Rtgtr2syMl.json"
lottie_json = load_lottieurl(lottie_url)

# --- 4. Main UI Logic based on Sidebar ---
if app_mode == "Dashboard":
    st.title("📊 Property Market Analysis Dashboard")
    st.markdown("## Overview of Housing Trends & Data Insights")
    st.markdown("---")
    
    # Lottie Animation
    col_lottie, col_title = st.columns([1, 4])
    with col_lottie:
        if lottie_json:
            st_lottie(lottie_json, height=100, key="house")

    # --- 6 Graphs Grid Layout ---
    st.subheader("Key Visualizations")
    
    # Graph Row 1
    g1, g2, g3 = st.columns(3)
    with g1:
        st.image("plot1.png", caption="Neighborhood Price Comparison")
    with g2:
        st.image("plot2.png", caption="Price vs Living Area")
    with g3:
        st.image("plot3.png", caption="Avg Price by House Style")

    # Graph Row 2
    g4, g5, g6 = st.columns(3)
    with g4:
        st.image("plot4.png", caption="House Style Distribution by Foundation")
    with g5:
        st.image("plot5.png", caption="Pairplot of Key Features")
    with g6:
        st.image("plot6.png", caption="Price Distribution by Quality")

elif app_mode == "Live Prediction":
    st.title("🔮 Live Property Price Prediction")
    st.markdown("### Input property details below to get an instant valuation.")
    st.markdown("---")
    
    # Input Fields Layout
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### Physical Attributes")
            gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=10000, value=1500)
            full_bath = st.slider("Full Bathrooms", 0, 5, 2)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2005)
            
        with c2:
            st.markdown("#### Location & Style")
            neighborhood = st.selectbox("Neighborhood", ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'NridgHt', 'Somerst'])
            house_style = st.selectbox("House Style", ['2Story', '1Story', '1.5Fin', 'SLvl'])
            foundation = st.selectbox("Foundation Type", ['PConc', 'CBlock', 'BrkTil', 'Wood'])
            
        with c3:
            st.markdown("#### Quality Attributes")
            kitchen_qual = st.selectbox("Kitchen Quality", ['Ex', 'Gd', 'TA', 'Fa'])
            central_air = st.radio("Central Air Conditioning", ['Y', 'N'])
            lot_shape = st.selectbox("Lot Shape", ['Reg', 'IR1', 'IR2', 'IR3'])
            
        submit_button = st.form_submit_button(label='🚀 Predict Price')

    # Prediction Logic
    if submit_button:
        with st.spinner('Calculating valuation...'):
            # Create DataFrame with exact column names from training
            input_data = pd.DataFrame({
                'GrLivArea': [gr_liv_area],
                'FullBath': [full_bath],
                'YearBuilt': [year_built],
                'Neighborhood': [neighborhood],
                'HouseStyle': [house_style],
                'Foundation': [foundation],
                'KitchenQual': [kitchen_qual],
                'CentralAir': [central_air],
                'LotShape': [lot_shape]
            })
            
            # Predict
            prediction = model.predict(input_data)[0]

            st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Split result area into two columns: Price on left, Animation on right
        res_col1, res_col2 = st.columns([2, 1])

        result_lottie_url = "https://lottie.host/428219b1-42a1-4fb9-8d80-9a61bbabe469/2n9FH1xgIu.json"
        result_lottie_json = load_lottieurl(result_lottie_url)
        
        with res_col1:
            st.markdown("### 💰 Estimated Property Value")
            st.success(f"# ${prediction:,.2f}")
            
        with res_col2:
            if result_lottie_json:
                # DIFFERENT ANIMATION KEY HERE
                st_lottie(result_lottie_json, height=150, key="result_animation")
        
        # 🎉 Lottie animation in prediction result
        st.balloons()