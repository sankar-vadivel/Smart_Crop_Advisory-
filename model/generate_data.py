"""
Synthetic crop recommendation dataset generator.

Models soil nutrient levels (N, P, K), temperature, humidity, soil pH, and
rainfall against a suitable crop, based on well-documented agronomic ranges
for each crop (typical values used in crop-recommendation literature/datasets).

Generates data for training a crop-recommendation classifier — the ML core
behind the Smart Crop Advisory System.
"""

import numpy as np
import pandas as pd

# Typical agronomic requirement ranges per crop: (N, P, K, temp°C, humidity%, ph, rainfall mm)
# Each value is (mean, std) used to sample realistic variation.
CROP_PROFILES = {
    "rice":       dict(N=(80, 12), P=(45, 10), K=(40, 10), temp=(25, 3), humidity=(82, 5), ph=(6.2, 0.4), rainfall=(220, 40)),
    "wheat":      dict(N=(70, 12), P=(50, 10), K=(35, 8),  temp=(18, 3), humidity=(60, 8), ph=(6.5, 0.4), rainfall=(90, 25)),
    "maize":      dict(N=(85, 15), P=(40, 10), K=(30, 8),  temp=(24, 3), humidity=(65, 8), ph=(6.3, 0.4), rainfall=(110, 30)),
    "cotton":     dict(N=(60, 10), P=(35, 8),  K=(45, 10), temp=(28, 3), humidity=(55, 8), ph=(6.8, 0.4), rainfall=(70, 20)),
    "sugarcane":  dict(N=(100, 15), P=(50, 10), K=(60, 12), temp=(27, 3), humidity=(78, 6), ph=(6.6, 0.4), rainfall=(180, 35)),
    "millet":     dict(N=(40, 8),  P=(25, 6),  K=(20, 6),  temp=(30, 3), humidity=(45, 8), ph=(6.9, 0.5), rainfall=(50, 15)),
    "groundnut":  dict(N=(35, 8),  P=(55, 10), K=(40, 8),  temp=(26, 3), humidity=(58, 8), ph=(6.4, 0.4), rainfall=(80, 20)),
    "pulses":     dict(N=(25, 6),  P=(45, 8),  K=(30, 6),  temp=(23, 3), humidity=(50, 8), ph=(6.7, 0.4), rainfall=(65, 18)),
}

SAMPLES_PER_CROP = 150


def generate_dataset(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for crop, p in CROP_PROFILES.items():
        for _ in range(SAMPLES_PER_CROP):
            row = {
                "N": max(0, rng.normal(*p["N"])),
                "P": max(0, rng.normal(*p["P"])),
                "K": max(0, rng.normal(*p["K"])),
                "temperature": rng.normal(*p["temp"]),
                "humidity": np.clip(rng.normal(*p["humidity"]), 10, 100),
                "ph": np.clip(rng.normal(*p["ph"]), 3.5, 9.5),
                "rainfall": max(0, rng.normal(*p["rainfall"])),
                "label": crop,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("crop_data.csv", index=False)
    print(f"Generated {len(df)} rows across {df['label'].nunique()} crops")
