import joblib, pandas as pd, sys

# Check if the file path was provided
if len(sys.argv) < 2:
    print("Usage: python predict.py <path_to_input_csv>")
    sys.exit(1)

# Load model from the models/ folder
bundle = joblib.load("models/crash_severity_model.joblib")

# Read the new data
df = pd.read_csv(sys.argv[1])

# Predict
proba = bundle["pipeline"].predict_proba(df[bundle["features"]])
p_non_casualty = proba[:, 0]  # class 0 = Non-casualty

# Assign labels based on the threshold
df["predicted_label"] = ["Non-casualty" if p >= bundle["threshold"] else "Casualty"
                          for p in p_non_casualty]
df["p_non_casualty"] = p_non_casualty

# Save to the processed folder
df.to_csv("data/processed/predictions.csv", index=False)

print("Predictions saved to data/processed/predictions.csv")
print(df[["predicted_label", "p_non_casualty"]].head())