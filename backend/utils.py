import joblib
import numpy as np
import pandas as pd

# === Load model and threshold ===
model = joblib.load("backend/model/xgb_model.joblib")
config = joblib.load("backend/model/model_config.joblib")
threshold = config["confidence_threshold"]

# === Define expected columns ===
EXPECTED_COLS = [
    'ID_Membre', 'Age', 'Sexe', 'Moyenne_Lycée', 'Autres_Clubs',
    'Projets_Realisés', 'Evaluation_Bureau', 'Soft_Skills',
    'Score_Entretien', 'Experience_Professionnelle', 'Indice_Engagement',
    'Cellule_Cellule Developpement Commercial', 'Cellule_Cellule Marketing',
    'Cellule_Cellule Mobile', 'Cellule_Cellule Qualité',
    'Cellule_Cellule RH', 'Cellule_Cellule Web',
    'Filiere_Encoded'
]

CELLULE_COLS = [
    'Cellule_Cellule Developpement Commercial',
    'Cellule_Cellule Marketing',
    'Cellule_Cellule Mobile',
    'Cellule_Cellule Qualité',
    'Cellule_Cellule RH',
    'Cellule_Cellule Web'
]

def transform_input(user: dict) -> pd.DataFrame:
    cellule = user.pop("Cellule")
    filiere = user.pop("Filiere")

    df = pd.DataFrame([user])
    df["ID_Membre"] = 0
    df["Filiere_Encoded"] = filiere

    # One-hot encoding for cellule
    for col in CELLULE_COLS:
        df[col] = 1 if col.endswith(cellule) else 0

    # Fill missing columns
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0

    return df[EXPECTED_COLS]

def predict_top_3(user_dict: dict):
    df = transform_input(user_dict)
    probs = model.predict_proba(df)

    if isinstance(probs, list):
        probs = np.mean(np.array(probs), axis=0)

    top_indices = np.argsort(probs[0])[::-1]
    results = []
    for idx in top_indices:
        if probs[0][idx] >= threshold:
            results.append({
                "formation_id": int(idx),
                "probability": round(float(probs[0][idx]), 4)
            })
        if len(results) == 3:
            break

    return results
