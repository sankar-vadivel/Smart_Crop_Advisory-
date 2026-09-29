"""
Smart Crop Advisory System — Flask backend.

Exposes:
  POST /api/recommend  -> ML-based crop recommendation from soil/climate inputs
  GET  /api/sensors     -> simulated IoT sensor reading (as if pulled from a
                            soil moisture / temperature / humidity sensor node)
"""
import os
import pickle
import random

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")

app = Flask(__name__)
CORS(app)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Irrigation advice is a simple rule engine layered on top of the sensor
# reading — a lightweight decision-support layer alongside the ML model.
def irrigation_advice(soil_moisture, temperature):
    if soil_moisture < 30:
        return "Irrigate now — soil moisture is low." if temperature < 35 \
            else "Irrigate now — low moisture and high heat increase crop stress."
    if soil_moisture < 55:
        return "Monitor closely — irrigate within the next 24 hours if no rainfall."
    return "No irrigation needed — soil moisture is sufficient."


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.get_json(force=True) or {}
    try:
        values = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": f"Provide numeric values for: {', '.join(FEATURES)}"}), 400

    X = pd.DataFrame([values], columns=FEATURES)
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]

    soil_moisture = float(data.get("soil_moisture", 50))
    temperature = values[3]

    return jsonify({
        "top_recommendation": ranked[0][0],
        "confidence": round(float(ranked[0][1]) * 100, 1),
        "alternatives": [
            {"crop": c, "confidence": round(float(p) * 100, 1)} for c, p in ranked[1:]
        ],
        "irrigation_advice": irrigation_advice(soil_moisture, temperature),
    })


@app.route("/api/sensors", methods=["GET"])
def sensors():
    """Simulated IoT sensor node reading (soil probe + weather module)."""
    return jsonify({
        "N": round(random.uniform(20, 110), 1),
        "P": round(random.uniform(15, 60), 1),
        "K": round(random.uniform(15, 65), 1),
        "temperature": round(random.uniform(15, 35), 1),
        "humidity": round(random.uniform(30, 90), 1),
        "ph": round(random.uniform(5.5, 8.0), 2),
        "rainfall": round(random.uniform(20, 250), 1),
        "soil_moisture": round(random.uniform(15, 85), 1),
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, port=5001)
