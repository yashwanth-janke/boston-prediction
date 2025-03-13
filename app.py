import json
import pickle
import os
import logging
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model and scaler
try:
    regmodel = pickle.load(open('regmodel.pkl', 'rb'))
    scalar = pickle.load(open('scaling.pkl', 'rb'))
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model files: {e}")

# Feature names for reference
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'Age', 
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        logger.info(f"API Prediction request received with data: {data}")
        
        # Validate input data
        for feature in feature_names:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
        
        # Transform data
        input_values = np.array(list(data.values())).reshape(1, -1)
        new_data = scalar.transform(input_values)
        
        # Make prediction
        output = regmodel.predict(new_data)
        logger.info(f"Prediction result: {output[0]}")
        
        # Format result
        result = {
            "prediction": float(output[0]),
            "formatted_price": f"${float(output[0]):,.2f}"
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in predict_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        data = [float(x) for x in request.form.values()]
        logger.info(f"Web form prediction request received")
        
        # Ensure correct number of features
        if len(data) != len(feature_names):
            return render_template("home.html", 
                                  prediction_text="Error: Invalid number of features provided")
        
        # Transform data
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        
        # Make prediction
        output = regmodel.predict(final_input)[0]
        logger.info(f"Prediction result: {output}")
        
        # Format the output as currency
        formatted_output = "${:,.2f}".format(output)
        
        return render_template("home.html", 
                              prediction_text=f"The House price prediction is {formatted_output}")
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return render_template("home.html", 
                             prediction_text=f"Error in prediction: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint for health checks"""
    return jsonify({"status": "healthy"})

@app.route('/sample_data', methods=['GET'])
def sample_data():
    """Provide sample data for testing"""
    sample = {
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
    return jsonify(sample)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)