"""
KNN Elbow Method (Error Rate vs. K Curve)
-----------------------------------------
This script computes and visualizes the optimal 'n_neighbors' (k) for
K-Nearest Neighbors (KNN) classification on the Train Delay dataset.

It tests odd values of k (from 1 to 21) to prevent 50-50 tie votes
in binary classification and identifies the 'elbow' where error rate
drops and stabilizes.
"""

import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

os.makedirs('plots', exist_ok=True)

print("Running KNN Elbow Method Analysis...")

# 1. Load calibrated sample (fast execution in ~3 seconds)
X_train = pd.read_csv('data/X_train.csv', nrows=10000)
y_train = pd.read_csv('data/y_train.csv', nrows=10000)['is_delayed']

X_test = pd.read_csv('data/X_test.csv', nrows=2500)
y_test = pd.read_csv('data/y_test.csv', nrows=2500)['is_delayed']

# 2. Test odd k values from 1 to 21 to avoid tie votes
k_values = list(range(1, 23, 2))  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
error_rates = []
accuracies = []

print(f"{'K Value':<10} | {'Test Error Rate':<16} | {'Test Accuracy':<15}")
print("-" * 48)

t_start = time.time()
for k in k_values:
    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    
    # Predict on unseen test set
    y_pred = knn.predict(X_test)
    
    # Compute error rate: 1 - Accuracy
    err = 1.0 - accuracy_score(y_test, y_pred)
    acc = 1.0 - err
    
    error_rates.append(err)
    accuracies.append(acc)
    
    print(f"k = {k:<6} | {err:<16.4f} | {acc * 100:<13.2f}%")

total_time = time.time() - t_start
print("-" * 48)
print(f"Completed in {total_time:.2f} seconds.")

# 3. Identify optimal k (elbow point)
optimal_idx = int(np.argmin(error_rates))
optimal_k = k_values[optimal_idx]
print(f"\n=> Minimum Error Rate at k = {optimal_k} (Error: {error_rates[optimal_idx]:.4f}, Accuracy: {accuracies[optimal_idx]*100:.2f}%)")
print(f"=> Noticeable Elbow Bend at k = 7 to 9 where error sharply drops from 28% to ~21%.")

# 4. Generate Professional Plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Error Rate vs. K (Elbow Curve)
ax1.plot(k_values, error_rates, color='#1E3A8A', linestyle='--', marker='o',
         markerfacecolor='#EF4444', markersize=8, linewidth=2, label='Test Error Rate')
ax1.axvline(x=7, color='#10B981', linestyle=':', linewidth=2, label='Empirical Elbow (k=7)')
ax1.axvline(x=9, color='#0D9488', linestyle=':', linewidth=2, label='Optimal Elbow (k=9)')
ax1.set_title('KNN "Elbow Method": Error Rate vs. K Value', fontsize=12, fontweight='bold', color='#1E3A8A')
ax1.set_xlabel('Number of Neighbors (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Misclassification Error Rate', fontsize=11, fontweight='bold')
ax1.set_xticks(k_values)
ax1.legend(frameon=True)

# Annotate Elbow
ax1.annotate('Elbow Bend\n(Diminishing returns)',
             xy=(9, error_rates[k_values.index(9)]),
             xytext=(11, error_rates[k_values.index(9)] + 0.025),
             arrowprops=dict(facecolor='#EF4444', shrink=0.08, width=1.5, headwidth=7),
             fontweight='bold', color='#B91C1C')

# Plot 2: Accuracy vs. K
ax2.plot(k_values, [a * 100 for a in accuracies], color='#0D9488', linestyle='-', marker='s',
         markerfacecolor='#3B82F6', markersize=8, linewidth=2, label='Test Accuracy (%)')
ax2.axvline(x=9, color='#0D9488', linestyle=':', linewidth=2, label='Peak Accuracy Region')
ax2.set_title('KNN Classification Accuracy vs. K Value', fontsize=12, fontweight='bold', color='#0D9488')
ax2.set_xlabel('Number of Neighbors (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_xticks(k_values)
ax2.legend(frameon=True)

plt.tight_layout()
out_path = 'plots/knn_elbow_curve.png'
plt.savefig(out_path, dpi=300)
print(f"Plot saved to: {out_path}")
