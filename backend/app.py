from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import requests
from dotenv import load_dotenv

# ================== ENV ==================
load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")

# ================== APP ==================
app = Flask(__name__)
CORS(app)

# ================== MODEL PATH ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')

# Load models
rf_reg = joblib.load(os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
le     = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

# ================== AQI BUCKET ==================
def get_aqi_bucket(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Satisfactory'
    elif aqi <= 200:
        return 'Moderate'
    elif aqi <= 300:
        return 'Poor'
    elif aqi <= 400:
        return 'Very Poor'
    else:
        return 'Severe'

BUCKET_COLOR = {
    'Good': '#00e400',
    'Satisfactory': '#92d050',
    'Moderate': '#ffff00',
    'Poor': '#ff7e00',
    'Very Poor': '#ff0000',
    'Severe': '#99004c'
}

# ================== ROUTE 1: GET AQI DATA ==================
@app.route('/get-aqi', methods=['POST'])
def get_aqi():
    try:
        data = request.json
        city = data['city']

        url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"
        response = requests.get(url)
        result = response.json()

        if result['status'] != 'ok':
            return jsonify({'error': 'City not found'})

        iaqi = result['data']['iaqi']

        def get_val(key):
            return iaqi.get(key, {}).get('v', 0)

        features = {
            'pm25': get_val('pm25'),
            'pm10': get_val('pm10'),
            'no2': get_val('no2'),
            'no': get_val('no'),
            'nox': get_val('nox'),
            'nh3': get_val('nh3'),
            'co': get_val('co'),
            'so2': get_val('so2'),
            'o3': get_val('o3'),
            'benzene': 0,
            'toluene': 0,
            'xylene': 0,
            'city_encoded': 0,
            'month': 6,
            'year': 2024,
            'season': 2,
            'pm25_7day': get_val('pm25'),
            'pm10_7day': get_val('pm10')
        }

        return jsonify(features)

    except Exception as e:
        return jsonify({'error': str(e)})

# ================== ROUTE 2: PREDICT ==================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = np.array([[
            data['pm25'], data['pm10'], data['no2'], data['no'],
            data['nox'], data['nh3'], data['co'], data['so2'],
            data['o3'], data['benzene'], data['toluene'], data['xylene'],
            data['city_encoded'], data['month'], data['year'],
            data['season'], data['pm25_7day'], data['pm10_7day']
        ]])

        features_scaled = scaler.transform(features)

        aqi_pred = float(rf_reg.predict(features_scaled)[0])
        bucket = get_aqi_bucket(aqi_pred)

        return jsonify({
            'aqi': round(aqi_pred, 1),
            'bucket': bucket,
            'color': BUCKET_COLOR[bucket]
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# ================== RUN ==================
if __name__ == '__main__':
    app.run(debug=True, port=5000)