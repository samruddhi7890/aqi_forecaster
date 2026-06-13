# AQI Forecaster — India Air Quality Prediction

A machine learning project that predicts the Air Quality Index (AQI) for major Indian cities using 5 years of CPCB pollution data, wrapped in a user-friendly web interface that fetches live sensor readings automatically.

---

## What it does

- Predicts **AQI value** (regression) and **AQI category** (Good / Moderate / Severe etc.) for any major Indian city
- Fetches **live pollutant data** automatically from the WAQI API — no manual input needed
- Displays **health advice**, a **colour-coded severity scale**, and an **interactive pollutant education panel**
- Built end-to-end: from raw CSV data → EDA → ML models → Flask API → browser UI

---

## Project Structure

```
aqi-forecaster/
├── data/
│   └── city_day.csv              # Kaggle dataset (not tracked by git)
├── notebooks/
│   └── aqi_project.ipynb         # Full EDA, preprocessing, model training
├── models/
│   ├── rf_regressor.pkl           # Random Forest Regressor (AQI value)
│   ├── rf_classifier.pkl          # Random Forest Classifier (AQI bucket)
│   ├── scaler.pkl                 # StandardScaler
│   └── label_encoder.pkl          # LabelEncoder for city names
├── backend/
│   └── app.py                    # Flask REST API
├── frontend/
│   └── index.html                # Single-page UI (HTML + CSS + JS)
└── README.md
```

---

## ML Pipeline

### Dataset
- **Source:** [India Air Quality Data — Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Size:** ~29,000 daily records across 26 Indian cities (2015–2020)
- **Features:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene

### Models Trained

| Task | Models | Best Result |
|------|--------|-------------|
| Regression (predict AQI value) | Linear Regression, Ridge, Random Forest | R² > 0.95 (RF) |
| Classification (predict AQI bucket) | Logistic Regression, Random Forest | Weighted F1 > 0.95 (RF) |

### Feature Engineering
- Temporal features: Month, Year, Season (4-bucket: Winter / Summer / Monsoon / Post-monsoon)
- City label encoding
- 7-day rolling averages for PM2.5 and PM10

---

## Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/aqi-forecaster.git
cd aqi-forecaster
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib flask flask-cors
```

### 3. Get the dataset
Download `city_day.csv` from [Kaggle](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) and place it in `data/`.

### 4. Train the models
Open and run all cells in `notebooks/aqi_project.ipynb`. This saves the four `.pkl` files into `models/`.

### 5. Get a free WAQI API token
Sign up at [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) — it's instant and free.  
Paste your token into `frontend/index.html`:
```javascript
const WAQI_TOKEN = 'your_token_here';
```

### 6. Start the Flask backend
```bash
cd backend
python app.py
# Running on http://localhost:5000
```

### 7. Open the frontend
In VS Code, right-click `frontend/index.html` → **Open with Live Server**  
Or just open it directly in your browser.

---

## API Reference

### `POST /predict`

**Request body (JSON):**
```json
{
  "pm25": 85.0,
  "pm10": 120.0,
  "no2": 35.0,
  "co": 1.2,
  "so2": 12.0,
  "o3": 40.0,
  "no": 0, "nox": 0, "nh3": 0,
  "benzene": 0, "toluene": 0, "xylene": 0,
  "city_encoded": 10,
  "month": 11, "year": 2024, "season": 3,
  "pm25_7day": 85.0, "pm10_7day": 120.0
}
```

**Response:**
```json
{
  "aqi": 218.4,
  "bucket": "Poor",
  "color": "#ff7e00"
}
```

---

## Supported Cities

Ahmedabad · Aizawl · Amaravati · Amritsar · Bengaluru · Bhopal · Brajrajnagar · Chandigarh · Chennai · Coimbatore · Delhi · Ernakulam · Gurugram · Guwahati · Hyderabad · Jaipur · Jodhpur · Kochi · Kolkata · Lucknow · Mumbai · Patna · Shillong · Talcher · Thiruvananthapuram · Visakhapatnam

---

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Data & EDA | Python, Pandas, NumPy, Matplotlib, Seaborn |
| ML | Scikit-learn (Random Forest, Ridge, Logistic Regression) |
| Model persistence | Joblib |
| Backend API | Flask, Flask-CORS |
| Frontend | HTML, CSS, JavaScript (Vanilla) |
| Live data | WAQI API (aqicn.org) |
| Dev tools | Jupyter Notebook, VS Code, Live Server, Git |
| Dataset | Kaggle — CPCB India Air Quality Data |

---

## AQI Scale (CPCB)

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0–50 | 🟢 Good | Minimal impact |
| 51–100 | 🟡 Satisfactory | Minor breathing discomfort for sensitive people |
| 101–200 | 🟠 Moderate | Breathing discomfort for asthma & lung disease patients |
| 201–300 | 🔴 Poor | Breathing discomfort for most people |
| 301–400 | 🟣 Very Poor | Respiratory illness on prolonged exposure |
| 401–500 | ⚫ Severe | Health alert — affects healthy people too |

---

## Key Files

| File | Purpose |
|------|---------|
| `notebooks/aqi_project.ipynb` | Full ML pipeline — EDA to saved models |
| `backend/app.py` | Flask API serving predictions |
| `frontend/index.html` | Complete single-page web app |
| `models/*.pkl` | Trained model artefacts |

---
## Acknowledgements

- [Rohan Rao](https://www.kaggle.com/rohanrao) for the Kaggle dataset
- [CPCB](https://cpcb.nic.in/) — Central Pollution Control Board of India
- [WAQI Project](https://waqi.info/) for the real-time air quality API
