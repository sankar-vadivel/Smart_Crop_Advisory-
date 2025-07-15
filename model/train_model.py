"""
Trains a RandomForest crop-recommendation classifier on soil/climate data
and saves the model for the Flask backend to load.
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from generate_data import generate_dataset

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train():
    df = generate_dataset()
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}\n")
    print(classification_report(y_test, preds))

    with open("crop_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to crop_model.pkl")


if __name__ == "__main__":
    train()
