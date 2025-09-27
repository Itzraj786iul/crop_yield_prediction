from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)

# Load all components
model_path = os.path.join(os.path.dirname(__file__), 'model')
model = joblib.load(os.path.join(model_path, 'crop_yield_model.pkl'))
base_scaler = joblib.load(os.path.join(model_path, 'base_scaler.pkl'))
interaction_scaler = joblib.load(os.path.join(model_path, 'interaction_scaler.pkl'))
label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.pkl'))
crop_to_category = joblib.load(os.path.join(model_path, 'crop_to_category.pkl'))
feature_cols = joblib.load(os.path.join(model_path, 'feature_cols.pkl'))

# Define numerical columns that need scaling
numerical_cols = ['Year', 'year_since_start', 'decade', 'AREA', 
                  'district_mean_yield', 'state_mean_yield', 'prev_year_yield', 'yield_deviation']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Year', 'State Code', 'Dist Code', 'State Name', 
                          'Dist Name', 'crop', 'AREA', 'district_mean_yield', 
                          'state_mean_yield', 'prev_year_yield']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data])
        
        # Calculate derived features
        input_df['year_since_start'] = input_df['Year'] - 1966
        input_df['decade'] = (input_df['Year'] // 10) * 10
        
        # Calculate yield deviation
        input_df['yield_deviation'] = input_df['prev_year_yield'] - input_df['district_mean_yield']
        
        # Get crop category
        input_df['crop_category'] = input_df['crop'].map(crop_to_category)
        
        # Handle unknown crops
        if input_df['crop_category'].isna().any():
            return jsonify({'error': f'Unknown crop: {data["crop"]}'}), 400
        
        # Encode categorical features
        for col in ['State Name', 'Dist Name', 'crop', 'crop_category']:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unknown categories
                try:
                    input_df[f'{col}_encoded'] = le.transform(input_df[col])
                except ValueError:
                    return jsonify({'error': f'Unknown {col}: {input_df[col].iloc[0]}'}), 400
        
        # Select features in the correct order
        features_df = input_df[feature_cols].copy()
        
        # Scale numerical features
        features_df[numerical_cols] = base_scaler.transform(features_df[numerical_cols])
        
        # Create interaction features
        features_df['yield_area_interaction'] = features_df['prev_year_yield'] * features_df['AREA']
        features_df['district_yield_interaction'] = features_df['district_mean_yield'] * features_df['prev_year_yield']
        features_df['state_yield_interaction'] = features_df['state_mean_yield'] * features_df['prev_year_yield']
        
        # Scale interaction features
        interaction_cols = ['yield_area_interaction', 'district_yield_interaction', 'state_yield_interaction']
        features_df[interaction_cols] = interaction_scaler.transform(features_df[interaction_cols])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'unit': 'Kg per ha',
            'input_data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/crops', methods=['GET'])
def get_crops():
    # Return list of supported crops
    crops = list(crop_to_category.keys())
    return jsonify({'crops': crops}), 200

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)