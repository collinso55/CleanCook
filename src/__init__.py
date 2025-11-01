import joblib
scaler = joblib.load("scaler.pkl")
print(scaler.feature_names_in_)
joblib.dump(scaler, "scaler_resaved.pkl", compress=3)
