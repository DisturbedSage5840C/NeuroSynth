import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("oasis_longitudinal.csv")

# Keep only clear classes for binary prediction.
df = df[df["Group"] != "Converted"]

features = ["Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]
df = df[features + ["Group"]].dropna()

le = LabelEncoder()
df["target"] = le.fit_transform(df["Group"])

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("Model saved!")
