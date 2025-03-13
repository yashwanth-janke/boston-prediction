import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }
    .block-container {
        max-width: 800px;
        margin: 0 auto;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 30px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        width: 100%;
        padding: 12px;
        margin-top: 20px;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .prediction-result {
        margin-top: 30px;
        padding: 15px;
        border-radius: 5px;
        background-color: #edf7ff;
        font-size: 18px;
        font-weight: 500;
        text-align: center;
    }
    .feature-description {
        font-size: 12px;
        color: #7f8c8d;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Boston House Price Prediction")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model_path = Path('regmodel.pkl')
        scaler_path = Path('scaling.pkl')
        
        if not model_path.exists() or not scaler_path.exists():
            st.error("Model files not found. Please ensure regmodel.pkl and scaling.pkl are in the app directory.")
            return None, None
        
        regmodel = pickle.load(open(model_path, 'rb'))
        scalar = pickle.load(open(scaler_path, 'rb'))
        logger.info("Model and scaler loaded successfully")
        return regmodel, scalar
    except Exception as e:
        logger.error(f"Error loading model files: {e}")
        st.error(f"Error loading model files: {e}")
        return None, None

regmodel, scalar = load_model_and_scaler()

# Feature names with descriptions
features = {
    'CRIM': 'Per capita crime rate by town',
    'ZN': 'Proportion of residential land zoned for large lots',
    'INDUS': 'Proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (1 if tract bounds river; 0 otherwise)',
    'NOX': 'Nitric oxides concentration (parts per 10 million)',
    'RM': 'Average number of rooms per dwelling',
    'Age': 'Proportion of owner-occupied units built prior to 1940',
    'DIS': 'Weighted distances to Boston employment centers',
    'RAD': 'Index of accessibility to radial highways',
    'TAX': 'Full-value property-tax rate per $10,000',
    'PTRATIO': 'Pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)² where Bk is the proportion of Black people',
    'LSTAT': '% lower status of the population'
}

# Provide sample data button
if st.button("Load Sample Data"):
    sample_data = {
        "CRIM": 0.00632,
        "ZN": 18.0,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.575,
        "Age": 65.2,
        "DIS": 4.0900,
        "RAD": 1,
        "TAX": 296,
        "PTRATIO": 15.3,
        "B": 396.90,
        "LSTAT": 4.98
    }
    st.session_state.update(sample_data)
    st.success("Sample data loaded!")

# Create form layout with 3 columns
st.write("Enter the property details below:")

    # Create a form
with st.form(key='house_price_form'):
    # Create 3 columns for layout
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store input values
    input_data = {}
    
    # Distribute fields across columns
    for i, (feature, description) in enumerate(features.items()):
        if i % 3 == 0:
            col = col1
        elif i % 3 == 1:
            col = col2
        else:
            col = col3
            
        with col:
            if feature == 'CHAS':
                # Special case for CHAS as it's a binary feature
                input_data[feature] = st.selectbox(
                    feature, 
                    options=[0, 1], 
                    help=description,
                    key=feature,
                    format_func=lambda x: f"{x} - {'Yes' if x == 1 else 'No'}"
                )
            elif feature in ['RAD', 'TAX']:
                # For integer features
                default_value = st.session_state.get(feature, 0)
                input_data[feature] = st.number_input(
                    feature, 
                    min_value=0, 
                    step=1,
                    help=description,
                    key=feature,
                    value=default_value
                )
            else:
                # For float features
                default_value = st.session_state.get(feature, 0.0)
                input_data[feature] = st.number_input(
                    feature, 
                    min_value=0.0, 
                    step=0.01,
                    help=description,
                    key=feature,
                    value=default_value
                )
            
            # Add description
            st.markdown(f'<div class="feature-description">{description}</div>', unsafe_allow_html=True)
    
    # Submit button
    submit_button = st.form_submit_button(label='Predict House Price')

# Make prediction if form is submitted
if submit_button:
    try:
        if regmodel is None or scalar is None:
            st.error("Model or scaler not loaded. Cannot make prediction.")
        else:
            # Convert input data to array
            input_values = np.array(list(input_data.values())).reshape(1, -1)
            
            # Scale the input data
            scaled_input = scalar.transform(input_values)
            
            # Make prediction
            prediction = regmodel.predict(scaled_input)[0]
            
            # Display prediction
            formatted_prediction = "${:,.2f}".format(prediction)
            st.markdown(f'<div class="prediction-result">The predicted house price is {formatted_prediction}</div>', unsafe_allow_html=True)
            
            # Log the prediction
            logger.info(f"Prediction made: {prediction}")
            
            # Show input data summary
            with st.expander("View Input Summary"):
                input_df = pd.DataFrame([input_data])
                st.dataframe(input_df)
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        st.error(f"Error making prediction: {e}")

# Add information about the dataset
with st.expander("About Boston Housing Dataset"):
    st.write("""
    The Boston Housing Dataset contains information about various houses in Boston through different parameters. 
    This dataset has 506 samples and 13 feature variables. The objective is to predict the value of houses.
    
    **Features:**
    - CRIM: Per capita crime rate by town
    - ZN: Proportion of residential land zoned for large lots
    - INDUS: Proportion of non-retail business acres per town
    - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    - NOX: Nitric oxides concentration (parts per 10 million)
    - RM: Average number of rooms per dwelling
    - AGE: Proportion of owner-occupied units built prior to 1940
    - DIS: Weighted distances to Boston employment centers
    - RAD: Index of accessibility to radial highways
    - TAX: Full-value property-tax rate per $10,000
    - PTRATIO: Pupil-teacher ratio by town
    - B: 1000(Bk - 0.63)² where Bk is the proportion of Black people
    - LSTAT: % lower status of the population
    
    **Target:**
    - MEDV: Median value of owner-occupied homes in $1000s
    """)

# Footer
st.markdown("---")
st.markdown("Boston House Price Prediction App | Built with Streamlit")