import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("Training production model for deployment...")

# Load dataset sample
df = pd.read_csv('data/ir_train.csv', nrows=100000)

# Drop leakage and ID columns
drop_cols = ['journey_id', 'train_number', 'departure_date', 'primary_delay_cause', 'delay_minutes', 'zone_abbr']
df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

target_col = 'is_delayed'
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

# Categorical columns
cat_cols = ['train_type', 'season', 'zone', 'source_station_category', 'destination_station_category', 'traction_type']

encoders = {}
cat_mappings = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    cat_mappings[col] = list(le.classes_)

# Calculate feature medians/modes for sensible defaults
feature_defaults = {}
for col in X.columns:
    if col in cat_cols:
        feature_defaults[col] = int(X[col].mode()[0])
    else:
        feature_defaults[col] = float(X[col].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting Classifier
print("Fitting Gradient Boosting Classifier...")
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Model Trained! Test Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")

# Save artifacts
os.makedirs('saved_model', exist_ok=True)
joblib.dump(model, 'saved_model/model.pkl')
joblib.dump(scaler, 'saved_model/scaler.pkl')
joblib.dump(encoders, 'saved_model/encoders.pkl')

meta = {
    'features': list(X.columns),
    'cat_cols': cat_cols,
    'cat_mappings': cat_mappings,
    'feature_defaults': feature_defaults,
    'metrics': {
        'accuracy': round(acc, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(auc, 4)
    }
}

with open('saved_model/metadata.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print("Saved model artifacts to saved_model/ directory:")
print(" - saved_model/model.pkl")
print(" - saved_model/scaler.pkl")
print(" - saved_model/encoders.pkl")
print(" - saved_model/metadata.json")
