import joblib, pandas as pd, sys

bundle = joblib.load("processed/crash_severity_model.joblib")
df = pd.read_csv(sys.argv[1])
proba = bundle["pipeline"].predict_proba(df[bundle["features"]])
p_non_casualty = proba[:, 0]  # class 0 = Non-casualty
df["predicted_label"] = ["Non-casualty" if p >= bundle["threshold"] else "Casualty"
                          for p in p_non_casualty]
df["p_non_casualty"] = p_non_casualty
df.to_csv("predictions.csv", index=False)
print(df[["predicted_label", "p_non_casualty"]].head())
