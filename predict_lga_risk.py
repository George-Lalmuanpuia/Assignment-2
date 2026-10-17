import joblib, pandas as pd, sys

bundle = joblib.load("processed/lga_risk_model.joblib")
df = pd.read_csv(sys.argv[1])
X = df[bundle["features"]].values
proba = bundle["pipeline"].predict_proba(X)[:, 1]  # P(high-risk)
df["predicted_high_risk"] = (proba >= bundle["threshold"]).astype(int)
df["p_high_risk"] = proba
df.to_csv("lga_risk_predictions.csv", index=False)
print(df[["LGA", "predicted_high_risk", "p_high_risk"]].head())
