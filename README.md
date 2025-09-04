# Smart Crop Advisory for People

**Smart India Hackathon project** — an IoT and Machine Learning–based advisory
system that recommends the most suitable crop and irrigation action from
real-time soil and weather sensor data.

## Abstract

Farmers frequently lack immediate access to data-driven guidance on which
crop best suits their soil and current climate conditions, leading to
suboptimal yields and inefficient water use. This project addresses that gap
with a Smart Crop Advisory System that ingests soil nutrient levels
(Nitrogen, Phosphorus, Potassium), soil pH, and climate parameters
(temperature, humidity, rainfall) — either entered manually or pulled from a
simulated IoT sensor node — and uses a trained Random Forest classifier to
recommend the most suitable crop, along with a confidence score and
alternative options. A lightweight rule-based irrigation layer additionally
flags whether irrigation is needed based on live soil moisture. The system is
exposed through a Flask REST API and a responsive web dashboard, giving
farmers or agricultural advisors a fast, low-cost decision-support tool.

## My role: Frontend Developer

I built the web dashboard — the sensor input form, the "pull live sensor
reading" flow, the recommendation display (top crop, confidence, alternatives),
and the irrigation advisory panel — consuming the ML-powered Flask API
built alongside it.

## Architecture

```
frontend/index.html  →  fetch()  →  backend/app.py (Flask API)  →  model/crop_model.pkl (RandomForest)
```

- **model/generate_data.py** — builds a training dataset of soil/climate
  profiles per crop based on typical agronomic requirement ranges
- **model/train_model.py** — trains and saves a RandomForest crop classifier
  (~92% test accuracy across 8 crops)
- **backend/app.py** — Flask API: `/api/recommend`, `/api/sensors` (simulated
  IoT reading), `/api/health`
- **frontend/index.html** — dashboard: sensor inputs, live-reading pull,
  recommendation + irrigation advice display

## Tech Stack

Python, Flask, scikit-learn, pandas, HTML, CSS, JavaScript (fetch API)

## Running it

```bash
pip install -r requirements.txt

# 1. Train the model (writes model/crop_model.pkl)
python model/generate_data.py
python model/train_model.py

# 2. Start the backend
python backend/app.py
# Flask runs on http://localhost:5000

# 3. Open the frontend
# Just open frontend/index.html in a browser
```

## Data note

`model/generate_data.py` generates a synthetic dataset based on
well-documented typical N-P-K and climate ranges per crop, so the project
trains and runs fully offline. Swap in a real agricultural sensor dataset
(e.g. a public crop-recommendation dataset) by replacing `crop_data.csv` with
the same column schema (`N, P, K, temperature, humidity, ph, rainfall, label`)
— no other code changes needed.

## Future Improvements

- Connect to real IoT hardware (soil moisture + NPK sensor probes) instead of
  simulated readings
- Add weather API integration for forecast-aware recommendations
- Multi-language support for regional farmer accessibility
- Historical yield tracking and feedback loop to improve model accuracy
