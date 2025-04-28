import joblib
#create file under model just to test path
config = joblib.load("formation-reco-app/backend/model/model_config.joblib")
print(config)
import joblib

model = joblib.load("formation-reco-app/backend/model/xgb_model.joblib")

print("=== Colonnes attendues par le modèle ===")
if hasattr(model, "get_booster"):
    print(model.get_booster().feature_names)  # pour XGBoost natif
elif hasattr(model, "feature_names_in_"):
    print(model.feature_names_in_)  # pour modèle scikit-learn
else:
    print("Impossible de détecter les noms de features.")
