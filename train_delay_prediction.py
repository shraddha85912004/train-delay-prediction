# ==============================================================================
# Machine Learning Mini Project
# Title: Predicting Train Delays Using Machine Learning Classifiers and Boosting Ensembles
# Group: 2 Members
# ==============================================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Set basic plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("Starting Train Delay Prediction Project...")

# ------------------------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------------------------
print("\n--- 1. Loading Data ---")
# Reading sample of data for faster training and demonstration
data_path = 'data/ir_train.csv'
df = pd.read_csv(data_path, nrows=60000)

print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 3 rows:")
print(df.head(3))

print("\nTarget class distribution:")
print(df['is_delayed'].value_counts())
print(f"Delay percentage: {df['is_delayed'].mean() * 100:.2f}%")

# ------------------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------------------
print("\n--- 2. Performing Simple EDA ---")

# Plot 1: Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='is_delayed', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Distribution of Target (0 = On-time, 1 = Delayed)')
plt.xlabel('Is Delayed (>15 mins)')
plt.ylabel('Count')
plt.xticks([0, 1], ['On-time (0)', 'Delayed (1)'])
plt.tight_layout()
plt.savefig('plots/target_distribution.png', dpi=120)
plt.close()

# Plot 2: Delay rate by Train Type
plt.figure(figsize=(10, 5))
train_type_delay = df.groupby('train_type')['is_delayed'].mean().sort_values(ascending=False)
train_type_delay.plot(kind='barh', color='steelblue', edgecolor='black')
plt.axvline(df['is_delayed'].mean(), color='red', linestyle='--', label=f'Avg Delay ({df["is_delayed"].mean():.1%})')
plt.title('Delay Rate by Train Type')
plt.xlabel('Proportion of Delayed Trains')
plt.legend()
plt.tight_layout()
plt.savefig('plots/delay_by_train_type.png', dpi=120)
plt.close()

# Plot 3: Delay rate by Season
plt.figure(figsize=(8, 4))
season_delay = df.groupby('season')['is_delayed'].mean().sort_values(ascending=False)
season_delay.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Delay Rate Across Seasons')
plt.ylabel('Proportion Delayed')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('plots/delay_by_season.png', dpi=120)
plt.close()

print("Saved EDA plots in plots/ folder.")

# ------------------------------------------------------------------------------
# 3. PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
print("\n--- 3. Preprocessing Data ---")

# Drop leakage columns (occur after trip) and ID columns
drop_cols = ['journey_id', 'train_number', 'departure_date', 'primary_delay_cause', 'delay_minutes', 'zone_abbr']
df_clean = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Separate features and target
X = df_clean.drop(columns=['is_delayed'])
y = df_clean['is_delayed']

# Categorical columns to encode
cat_cols = ['train_type', 'season', 'zone', 'source_station_category', 'destination_station_category', 'traction_type']
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ------------------------------------------------------------------------------
# 4. TRAINING 6 MACHINE LEARNING MODELS (4 Baselines + 2 Ensembles)
# ------------------------------------------------------------------------------
print("\n--- 4. Training 6 Shortlisted Models ---")

# 4 baseline models + 2 easy ensembles (AdaBoost and Gradient Boosting, no Random Forest)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    "AdaBoost (Ensemble 1)": AdaBoostClassifier(n_estimators=80, random_state=42),
    "Gradient Boosting (Ensemble 2)": GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)
}

results = []
roc_curves = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"Training {name}...")
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0
    
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    
    # Predict probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr, auc)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    results.append({
        "Model": name,
        "Type": "Ensemble (Boosting)" if "Ensemble" in name else "Individual Classifier",
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Train Time (s)": round(train_time, 2)
    })

# Convert to DataFrame
results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
results_df.to_csv("results/model_comparison_results.csv", index=False)

print("\n" + "=" * 70)
print("              MODEL EVALUATION RESULTS TABLE")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

# ------------------------------------------------------------------------------
# 5. GENERATE EVALUATION PLOTS
# ------------------------------------------------------------------------------
print("\n--- 5. Generating Evaluation Plots ---")

# Plot 4: Combined ROC Curves
plt.figure(figsize=(9, 7))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison of 6 Models')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=120)
plt.close()

# Plot 5: Model Comparison Bar Chart
plt.figure(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.35
plt.bar(x - width/2, results_df['Accuracy'], width, label='Accuracy', color='#3498db', edgecolor='black')
plt.bar(x + width/2, results_df['F1-Score'], width, label='F1-Score', color='#2ecc71', edgecolor='black')
plt.xticks(x, results_df['Model'], rotation=25, ha='right')
plt.ylabel('Score')
plt.title('Accuracy and F1-Score Across Models')
plt.ylim(0.6, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig('plots/model_comparison_barchart.png', dpi=120)
plt.close()

# Plot 6: Confusion Matrices (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for idx, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(name, fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
plt.suptitle('Confusion Matrices for All 6 Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('plots/confusion_matrices.png', dpi=120)
plt.close()

# Plot 7: Feature Importance (from Gradient Boosting)
gb_model = models["Gradient Boosting (Ensemble 2)"]
feat_importances = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(12)

plt.figure(figsize=(9, 5))
feat_importances.plot(kind='barh', color='#2c3e50', edgecolor='black')
plt.title('Top 12 Features (Gradient Boosting)')
plt.xlabel('Relative Feature Importance')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=120)
plt.close()

best_model = results_df.iloc[0]
print(f"\nBest Model: {best_model['Model']} with ROC-AUC: {best_model['ROC-AUC']}")
print("Done! All results and plots saved successfully.")
