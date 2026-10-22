import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load saved model artifacts
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_model')
model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
encoders = joblib.load(os.path.join(MODEL_DIR, 'encoders.pkl'))

with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

features = metadata['features']
cat_mappings = metadata['cat_mappings']
defaults = metadata['feature_defaults']
metrics = metadata['metrics']

@app.route('/')
def home():
    return render_template(
        'index.html',
        train_types=cat_mappings['train_type'],
        seasons=cat_mappings['season'],
        zones=cat_mappings['zone'],
        station_cats=cat_mappings['source_station_category'],
        metrics=metrics,
        result=None
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form inputs
        train_type = request.form.get('train_type', 'Mail/Express')
        season = request.form.get('season', 'Monsoon')
        zone = request.form.get('zone', 'Northern Railway (NR)')
        source_cat = request.form.get('source_station_category', 'A')
        dest_cat = request.form.get('destination_station_category', 'B')
        traction_type = request.form.get('traction_type', 'Electric (25kV AC)')
        
        distance_km = float(request.form.get('distance_km', 500))
        scheduled_hours = float(request.form.get('scheduled_travel_hours', 8.5))
        num_stops = int(request.form.get('num_scheduled_stops', 10))
        departure_hour = int(request.form.get('departure_hour', 14))
        ontime_pct = float(request.form.get('route_historical_ontime_pct', 65.0))
        
        # Binary flags
        late_rake = 1 if request.form.get('late_incoming_rake') == '1' else 0
        track_doubled = 1 if request.form.get('track_doubled') == '1' else 0
        is_weekend = 1 if request.form.get('is_weekend') == '1' else 0
        is_monsoon = 1 if season == 'Monsoon' else 0
        is_night = 1 if departure_hour in [22, 23, 0, 1, 2, 3, 4] else 0
        is_peak = 1 if departure_hour in [6, 7, 8, 17, 18, 19, 20] else 0

        # Construct full 38-feature vector starting with sensible defaults
        input_data = defaults.copy()
        
        # Update with user inputs
        input_data['distance_km'] = distance_km
        input_data['scheduled_travel_hours'] = scheduled_hours
        input_data['num_scheduled_stops'] = num_stops
        input_data['departure_hour'] = departure_hour
        input_data['route_historical_ontime_pct'] = ontime_pct
        input_data['late_incoming_rake'] = late_rake
        input_data['track_doubled'] = track_doubled
        input_data['is_weekend'] = is_weekend
        input_data['is_monsoon_season'] = is_monsoon
        input_data['is_night_departure'] = is_night
        input_data['is_peak_hour'] = is_peak
        
        # Season severity proxy
        if season == 'Monsoon':
            input_data['season_severity_score'] = 0.85
        elif season == 'Winter/Fog':
            input_data['season_severity_score'] = 0.70
            input_data['is_fog_risk'] = 1
            input_data['fog_risk_score'] = 0.65
        else:
            input_data['season_severity_score'] = 0.35
            input_data['is_fog_risk'] = 0
            input_data['fog_risk_score'] = 0.0

        # Encode categorical inputs
        input_data['train_type'] = int(encoders['train_type'].transform([train_type])[0])
        input_data['season'] = int(encoders['season'].transform([season])[0])
        input_data['zone'] = int(encoders['zone'].transform([zone])[0])
        input_data['source_station_category'] = int(encoders['source_station_category'].transform([source_cat])[0])
        input_data['destination_station_category'] = int(encoders['destination_station_category'].transform([dest_cat])[0])
        input_data['traction_type'] = int(encoders['traction_type'].transform([traction_type])[0])

        # Convert to DataFrame in exact training order
        row_df = pd.DataFrame([input_data])[features]
        
        # Scale features
        row_scaled = scaler.transform(row_df)
        
        # Predict
        pred = model.predict(row_scaled)[0]
        prob_delay = model.predict_proba(row_scaled)[0][1] * 100
        
        # Identify key risk factors for this prediction
        risk_factors = []
        if late_rake == 1:
            risk_factors.append("Late Incoming Rake: Previous train was already delayed.")
        if ontime_pct < 60:
            risk_factors.append(f"Low Route Reliability: Past punctuality on this route is only {ontime_pct}%.")
        if season in ['Monsoon', 'Winter/Fog']:
            risk_factors.append(f"Adverse Weather: High seasonal risk ({season}).")
        if track_doubled == 0:
            risk_factors.append("Single Track: High risk of crossing halts and waiting.")
        if is_peak == 1:
            risk_factors.append("Peak Hours: Elevated corridor congestion.")
        if not risk_factors:
            risk_factors.append("Standard operational conditions.")

        result = {
            'prediction': 'Delayed (>15 mins)' if pred == 1 else 'On-Time (≤15 mins)',
            'is_delayed': int(pred),
            'probability': round(prob_delay, 1),
            'risk_factors': risk_factors,
            'user_inputs': {
                'train_type': train_type,
                'season': season,
                'zone': zone,
                'distance_km': distance_km,
                'ontime_pct': ontime_pct
            }
        }
        
        return render_template(
            'index.html',
            train_types=cat_mappings['train_type'],
            seasons=cat_mappings['season'],
            zones=cat_mappings['zone'],
            station_cats=cat_mappings['source_station_category'],
            metrics=metrics,
            result=result
        )
        
    except Exception as e:
        return render_template(
            'index.html',
            train_types=cat_mappings['train_type'],
            seasons=cat_mappings['season'],
            zones=cat_mappings['zone'],
            station_cats=cat_mappings['source_station_category'],
            metrics=metrics,
            error=str(e),
            result=None
        )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for programmatic testing."""
    data = request.get_json(force=True)
    input_data = defaults.copy()
    
    for k, v in data.items():
        if k in input_data:
            if k in encoders:
                input_data[k] = int(encoders[k].transform([str(v)])[0])
            else:
                input_data[k] = float(v)
                
    row_df = pd.DataFrame([input_data])[features]
    row_scaled = scaler.transform(row_df)
    
    pred = int(model.predict(row_scaled)[0])
    prob_delay = float(model.predict_proba(row_scaled)[0][1])
    
    return jsonify({
        'is_delayed': pred,
        'status': 'Delayed (>15 mins)' if pred == 1 else 'On-Time',
        'delay_probability': round(prob_delay, 4),
        'model': 'Gradient Boosting Classifier'
    })

if __name__ == '__main__':
    print("Starting Train Delay Prediction Web App...")
    print("Access the application at: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
