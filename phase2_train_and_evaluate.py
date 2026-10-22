"""
=============================================================================
  ML Mini Project – Phase 2: Model Implementation & Evaluation
  Title: Predictive Modeling of Transit Delays in Complex Railway Networks:
         A Comparative Study of Classical Classifiers and Adaptive Ensemble Architectures
=============================================================================
  Shortlisted Models (6 Total, including 2 Easy Ensembles - No Random Forest):
    1. Logistic Regression          [Linear Baseline]
    2. Decision Tree Classifier     [Tree-based Single Model]
    3. Gaussian Naive Bayes         [Probabilistic Classifier]
    4. K-Nearest Neighbors (KNN)    [Instance-based Classifier]
    5. AdaBoost Classifier          [ENSEMBLE 1: Adaptive Boosting]
    6. Gradient Boosting Classifier [ENSEMBLE 2: Gradient Boosting]
=============================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 75)
print("  ML MINI PROJECT - PHASE 2: 6 MODELS (WITH ADABOOST & GRADIENT BOOSTING)")
print("=" * 75)

# ---------------------------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# ---------------------------------------------------------------------------
print("\n[Step 1] Loading preprocessed data...")
t0 = time.time()

SAMPLE_SIZE_TRAIN = 80000
SAMPLE_SIZE_TEST = 20000

print(f"Reading dataset samples (Train: {SAMPLE_SIZE_TRAIN:,}, Test: {SAMPLE_SIZE_TEST:,})...")

X_train = pd.read_csv('data/X_train.csv', nrows=SAMPLE_SIZE_TRAIN)
y_train = pd.read_csv('data/y_train.csv', nrows=SAMPLE_SIZE_TRAIN)['is_delayed']

X_test = pd.read_csv('data/X_test.csv', nrows=SAMPLE_SIZE_TEST)
y_test = pd.read_csv('data/y_test.csv', nrows=SAMPLE_SIZE_TEST)['is_delayed']

print(f"Loaded in {time.time() - t0:.2f}s")
print(f"X_train shape: {X_train.shape}, y_train class balance: {dict(y_train.value_counts(normalize=True).round(3))}")
print(f"X_test shape : {X_test.shape}, y_test class balance : {dict(y_test.value_counts(normalize=True).round(3))}")

# ---------------------------------------------------------------------------
# 2. DEFINE THE 6 SHORTLISTED MODELS (4 Individual + 2 Easy Ensembles)
# ---------------------------------------------------------------------------
print("\n[Step 2] Initializing 6 Shortlisted Models...")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42, solver='lbfgs'
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_split=20, random_state=42
    ),
    "Gaussian Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(
        n_neighbors=9, n_jobs=-1
    ),
    "AdaBoost (Ensemble 1)": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.8, random_state=42
    ),
    "Gradient Boosting (Ensemble 2)": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
}

print(f"Configured {len(models)} models:")
for idx, (name, m) in enumerate(models.items(), 1):
    m_type = "Ensemble (Boosting)" if "Ensemble" in name else "Individual Classifier"
    print(f"  {idx}. {name:<32} -> [{m_type}]")

# ---------------------------------------------------------------------------
# 3. TRAIN & EVALUATE ALL MODELS
# ---------------------------------------------------------------------------
print("\n[Step 3] Training and evaluating models...")

results_list = []
roc_curves = {}
confusion_matrices = {}

for name, model in models.items():
    print(f"\n--- Training: {name} ---")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    
    start_pred = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_pred
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)
    confusion_matrices[name] = confusion_matrix(y_test, y_pred)
    
    m_type = "Ensemble (Boosting)" if "Ensemble" in name else "Individual Classifier"
    results_list.append({
        "Model": name,
        "Type": m_type,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "Train Time (s)": round(train_time, 2),
        "Pred Time (s)": round(pred_time, 2)
    })
    
    print(f"  Accuracy : {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(f"  Train Time: {train_time:.2f}s | Pred Time: {pred_time:.2f}s")

# ---------------------------------------------------------------------------
# 4. TABULATE & SAVE RESULTS
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results_list)
results_df = results_df.sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
results_df.to_csv("results/model_comparison_results.csv", index=False)

print("\n" + "=" * 80)
print("                      MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)
print("[OK] Results saved to 'results/model_comparison_results.csv'")

# ---------------------------------------------------------------------------
# 5. GENERATE COMPREHENSIVE EVALUATION PLOTS
# ---------------------------------------------------------------------------
print("\n[Step 5] Generating Comparison Visualizations...")

# Plot 1: Combined ROC Curves
plt.figure(figsize=(10, 8))
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves Comparison Across 6 Shortlisted Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig("plots/13_roc_curves_comparison.png", dpi=150)
plt.close()
print("[OK] Saved: plots/13_roc_curves_comparison.png")

# Plot 2: Model Comparison Bar Chart (Metrics Comparison)
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
ax = results_df.set_index("Model")[metrics_to_plot].plot(
    kind="bar", figsize=(14, 7), width=0.8, edgecolor="black"
)
plt.title("Performance Metrics Comparison Across 6 Shortlisted Models", fontsize=15, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=20, ha='right', fontsize=11)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
for p in ax.patches:
    h = p.get_height()
    if h > 0.1:
        ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', fontsize=8, color='white', rotation=90, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/14_model_metrics_barchart.png", dpi=150)
plt.close()
print("[OK] Saved: plots/14_model_metrics_barchart.png")

# Plot 3: Confusion Matrices (2x3 Grid)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, cm) in enumerate(confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=axes[idx],
                cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
    axes[idx].set_title(f"{name}", fontsize=12, fontweight='bold')
    axes[idx].set_xlabel("Predicted Label (0=OnTime, 1=Delayed)", fontsize=10)
    axes[idx].set_ylabel("True Label (0=OnTime, 1=Delayed)", fontsize=10)

plt.suptitle("Confusion Matrices for 6 Shortlisted Models", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("plots/15_confusion_matrices.png", dpi=150)
plt.close()
print("[OK] Saved: plots/15_confusion_matrices.png")

# Plot 4: Feature Importance (from Gradient Boosting Ensemble)
gb_model = models["Gradient Boosting (Ensemble 2)"]
feature_names = X_train.columns
importances = gb_model.feature_importances_
feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(15)

plt.figure(figsize=(12, 7))
plt.barh(feat_df["Feature"], feat_df["Importance"], color="navy", edgecolor="black")
plt.title("Top 15 Most Important Features (Gradient Boosting Ensemble)", fontsize=14, fontweight='bold')
plt.xlabel("Relative Feature Importance", fontsize=12)
for i, v in enumerate(feat_df["Importance"]):
    plt.text(v + 0.002, i, f"{v:.4f}", va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("plots/16_ensemble_feature_importance.png", dpi=150)
plt.close()
print("[OK] Saved: plots/16_ensemble_feature_importance.png")

# Plot 5: Training Time vs Performance Trade-off
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    results_df["Train Time (s)"], results_df["ROC-AUC"],
    s=results_df["F1-Score"]*400, c=results_df["Accuracy"],
    cmap='viridis', edgecolors='black', alpha=0.85
)
for _, row in results_df.iterrows():
    plt.annotate(
        row["Model"], (row["Train Time (s)"], row["ROC-AUC"]),
        textcoords="offset points", xytext=(8, 5), fontsize=10, fontweight='bold'
    )
plt.colorbar(scatter, label='Accuracy')
plt.xlabel("Training Time (seconds)", fontsize=12)
plt.ylabel("ROC-AUC Score", fontsize=12)
plt.title("Model Efficiency vs Accuracy Trade-off (Bubble Size = F1 Score)", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("plots/17_time_vs_performance.png", dpi=150)
plt.close()
print("[OK] Saved: plots/17_time_vs_performance.png")

best_model = results_df.iloc[0]
print("\n" + "=" * 75)
print(f"  BEST MODEL SELECTED: {best_model['Model']}")
print(f"  Type      : {best_model['Type']}")
print(f"  ROC-AUC   : {best_model['ROC-AUC']:.4f}")
print(f"  Accuracy  : {best_model['Accuracy']:.4f}")
print(f"  F1-Score  : {best_model['F1-Score']:.4f}")
print("=" * 75)
print("\n[DONE] All training, evaluation, and visualizations completed successfully!")
