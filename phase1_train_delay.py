"""
=============================================================================
  ML Mini Project – Phase 1
  Train Delay Prediction (Indian Railways)
  Dataset: Indian Railways – Predict Train Delay (Kaggle Competition)
=============================================================================
  Team Size: 2
  
  Phase 1 Checklist:
    1. [OK] Identify the dataset
    2. [OK] List the metadata of the dataset
    3. [OK] Define the problem statement and objectives
    4. [OK] Perform Exploratory Data Analysis (EDA)
    5. [OK] Preprocessing pipeline specific to the data
    6. [OK] Shortlist at least 6 models (2+ ensembles)
=============================================================================
"""

# ============================================================================
# 0. IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("  ML MINI PROJECT – PHASE 1: TRAIN DELAY PREDICTION")
print("=" * 70)

# ============================================================================
# 1. DATASET IDENTIFICATION
# ============================================================================
print("\n" + "=" * 70)
print("  1. DATASET IDENTIFICATION")
print("=" * 70)

print("""
Dataset Name  : Indian Railways – Predict Train Delay
Source         : Kaggle Competition
URL           : https://www.kaggle.com/competitions/133990
Domain        : Transportation / Indian Railways
Task Type     : Binary Classification
Period        : 2018 – 2024
Records       : 1,500,000 (train) + test set
Features      : 42 features + 1 target + 2 metadata columns
Target        : is_delayed (1 = train arrived >15 min late, 0 = on time)
""")

# Load data
DATA_PATH = r'data\ir_train.csv'
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================================
# 2. METADATA OF THE DATASET
# ============================================================================
print("\n" + "=" * 70)
print("  2. METADATA OF THE DATASET")
print("=" * 70)

# 2a. Basic Info
print("\n--- 2a. Shape & Memory ---")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")
print(f"Memory  : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# 2b. Column info
print("\n--- 2b. Column Names, Types & Non-Null Counts ---")
print(df.info())

# 2c. Data types summary
print("\n--- 2c. Data Types Summary ---")
print(df.dtypes.value_counts())

# 2d. Null values
print("\n--- 2d. Missing Values ---")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("[OK] No missing values found in any column!")
else:
    print(null_counts[null_counts > 0])

# 2e. Feature categories
print("\n--- 2e. Feature Groups (from Data Dictionary) ---")
feature_groups = {
    "Time/Temporal": ['year', 'month', 'day_of_week', 'departure_hour', 
                      'is_weekend', 'is_night_departure', 'is_peak_hour',
                      'is_festival_season', 'season'],
    "Geography/Zone": ['zone', 'zone_abbr', 'source_station_category',
                       'destination_station_category', 'zone_congestion_index',
                       'zone_fog_index'],
    "Route": ['distance_km', 'num_scheduled_stops', 'scheduled_travel_hours',
              'track_doubled', 'is_hdn_route', 'traction_type', 
              'is_electrified', 'psr_count', 'is_circular_route'],
    "Weather": ['is_monsoon_season', 'is_fog_risk', 'fog_risk_score',
                'season_severity_score'],
    "Rolling Stock": ['loco_age_years', 'coach_age_years', 'has_lhb_coaches',
                      'is_rake_shared', 'maintenance_score'],
    "Operations": ['seat_utilisation_pct', 'is_overloaded', 
                   'late_incoming_rake', 'is_special_train',
                   'route_historical_ontime_pct']
}
for group, cols in feature_groups.items():
    print(f"\n  {group} ({len(cols)} features):")
    for c in cols:
        print(f"    - {c} ({df[c].dtype})")

# ============================================================================
# 3. PROBLEM STATEMENT & OBJECTIVES
# ============================================================================
print("\n" + "=" * 70)
print("  3. PROBLEM STATEMENT & OBJECTIVES")
print("=" * 70)

print("""
PROBLEM STATEMENT:
------------------
Indian Railways, one of the largest railway networks in the world, faces 
significant challenges with train punctuality. Delays affect millions of 
passengers daily and lead to operational inefficiencies.

The goal is to build a machine learning model that predicts whether a train 
will arrive MORE THAN 15 MINUTES LATE at its destination, based on features 
available BEFORE departure.

OBJECTIVES:
-----------
  1. Explore and understand the patterns in 1.5M historical train journeys
  2. Identify key factors contributing to train delays
  3. Build preprocessing pipelines to handle categorical, numerical and 
     temporal features
  4. Compare 6+ classification models (including 2+ ensemble methods)
  5. Evaluate models using AUC-ROC as the primary metric
  6. Select the best-performing model for deployment

TARGET VARIABLE:
  is_delayed = 1 → Train arrived >15 minutes late
  is_delayed = 0 → Train arrived on time or within 15 minutes

TYPE: Binary Classification (Supervised Learning)

KEY METRIC: AUC-ROC Score
""")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 70)
print("  4. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 70)

# --- 4a. Target Variable Distribution ---
print("\n--- 4a. Target Variable Distribution ---")
target_dist = df['is_delayed'].value_counts()
print(target_dist)
print(f"\nDelay Rate: {df['is_delayed'].mean()*100:.2f}%")
print(f"Class Imbalance Ratio: {target_dist[1]/target_dist[0]:.2f}:1 (Delayed:On-Time)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Bar plot
colors = ['#2ecc71', '#e74c3c']
target_dist.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Is Delayed (0=On Time, 1=Delayed)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['On Time (0)', 'Delayed (1)'], rotation=0)
for i, v in enumerate(target_dist.values):
    axes[0].text(i, v + 10000, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                ha='center', fontweight='bold')
# Pie chart
axes[1].pie(target_dist.values, labels=['On Time', 'Delayed >15min'],
            autopct='%1.1f%%', colors=colors, startangle=90,
            explode=(0.05, 0), shadow=True)
axes[1].set_title('Delay Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/01_target_distribution.png")

# --- 4b. Numerical Features Distribution ---
print("\n--- 4b. Numerical Features Distribution ---")
numerical_cols = ['distance_km', 'num_scheduled_stops', 'scheduled_travel_hours',
                  'loco_age_years', 'coach_age_years', 'maintenance_score',
                  'seat_utilisation_pct', 'route_historical_ontime_pct',
                  'fog_risk_score', 'zone_congestion_index', 'psr_count',
                  'season_severity_score']

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    df[col].hist(bins=50, ax=axes[i], color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
    axes[i].legend(fontsize=8)
plt.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/02_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/02_numerical_distributions.png")

# --- 4c. Delay by Train Type ---
print("\n--- 4c. Delay Rate by Train Type ---")
delay_by_type = df.groupby('train_type')['is_delayed'].mean().sort_values(ascending=False)
print(delay_by_type)

fig, ax = plt.subplots(figsize=(14, 6))
delay_by_type.plot(kind='barh', ax=ax, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(delay_by_type))),
                   edgecolor='black')
ax.set_title('Delay Rate by Train Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Delay Rate (Proportion)')
ax.axvline(df['is_delayed'].mean(), color='red', linestyle='--', linewidth=2, label=f'Overall: {df["is_delayed"].mean():.2%}')
ax.legend(fontsize=12)
for i, (val, name) in enumerate(zip(delay_by_type.values, delay_by_type.index)):
    ax.text(val + 0.005, i, f'{val:.1%}', va='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/03_delay_by_train_type.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/03_delay_by_train_type.png")

# --- 4d. Delay by Season ---
print("\n--- 4d. Delay Rate by Season ---")
delay_by_season = df.groupby('season')['is_delayed'].mean().sort_values(ascending=False)
print(delay_by_season)

fig, ax = plt.subplots(figsize=(10, 5))
delay_by_season.plot(kind='bar', ax=ax, color=['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db', '#9b59b6'],
                     edgecolor='black')
ax.set_title('Delay Rate by Season', fontsize=14, fontweight='bold')
ax.set_ylabel('Delay Rate')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
for i, v in enumerate(delay_by_season.values):
    ax.text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/04_delay_by_season.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/04_delay_by_season.png")

# --- 4e. Delay by Hour of Day ---
print("\n--- 4e. Delay Rate by Departure Hour ---")
delay_by_hour = df.groupby('departure_hour')['is_delayed'].mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(delay_by_hour.index, delay_by_hour.values, color='steelblue', edgecolor='black', alpha=0.8)
ax.plot(delay_by_hour.index, delay_by_hour.values, 'ro-', markersize=6)
ax.set_title('Delay Rate by Departure Hour', fontsize=14, fontweight='bold')
ax.set_xlabel('Departure Hour (0-23)')
ax.set_ylabel('Delay Rate')
ax.set_xticks(range(24))
ax.axhline(df['is_delayed'].mean(), color='red', linestyle='--', alpha=0.5, label='Overall Average')
ax.legend()
plt.tight_layout()
plt.savefig('plots/05_delay_by_hour.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/05_delay_by_hour.png")

# --- 4f. Delay by Zone ---
print("\n--- 4f. Delay Rate by Railway Zone ---")
delay_by_zone = df.groupby('zone_abbr')['is_delayed'].mean().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
delay_by_zone.plot(kind='bar', ax=ax, color=plt.cm.coolwarm(np.linspace(0.1, 0.9, len(delay_by_zone))),
                   edgecolor='black')
ax.set_title('Delay Rate by Railway Zone', fontsize=14, fontweight='bold')
ax.set_ylabel('Delay Rate')
ax.axhline(df['is_delayed'].mean(), color='black', linestyle='--', alpha=0.7, label='Overall Average')
ax.legend()
plt.tight_layout()
plt.savefig('plots/06_delay_by_zone.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/06_delay_by_zone.png")

# --- 4g. Correlation Heatmap ---
print("\n--- 4g. Correlation Heatmap ---")
numeric_df = df.select_dtypes(include=[np.number])
# Remove leakage columns
numeric_df = numeric_df.drop(columns=['delay_minutes'], errors='ignore')

corr = numeric_df.corr()
# Show top correlations with target
target_corr = corr['is_delayed'].drop('is_delayed').sort_values(key=abs, ascending=False)
print("\nTop 15 correlations with target (is_delayed):")
print(target_corr.head(15))

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/07_correlation_heatmap.png")

# --- 4h. Top correlated features bar chart ---
fig, ax = plt.subplots(figsize=(12, 8))
target_corr.head(20).plot(kind='barh', ax=ax, color=['#e74c3c' if x > 0 else '#3498db' for x in target_corr.head(20)],
                          edgecolor='black')
ax.set_title('Top 20 Features Correlated with Train Delay', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('plots/08_top_correlations.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/08_top_correlations.png")

# --- 4i. Delay by Station Category ---
print("\n--- 4i. Delay by Station Category (Source & Destination) ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(['source_station_category', 'destination_station_category']):
    delay_by_cat = df.groupby(col)['is_delayed'].mean().sort_index()
    delay_by_cat.plot(kind='bar', ax=axes[i], color='teal', edgecolor='black', alpha=0.8)
    axes[i].set_title(f'Delay Rate by {col}', fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Delay Rate')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig('plots/09_delay_by_station_category.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/09_delay_by_station_category.png")

# --- 4j. Primary Delay Causes ---
print("\n--- 4j. Primary Delay Causes (Train set only) ---")
delay_causes = df[df['is_delayed'] == 1]['primary_delay_cause'].value_counts()
print(delay_causes)

fig, ax = plt.subplots(figsize=(12, 6))
delay_causes.plot(kind='barh', ax=ax, color=plt.cm.Reds(np.linspace(0.3, 0.9, len(delay_causes))),
                  edgecolor='black')
ax.set_title('Primary Causes of Train Delay', fontsize=14, fontweight='bold')
ax.set_xlabel('Count')
plt.tight_layout()
plt.savefig('plots/10_delay_causes.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/10_delay_causes.png")

# --- 4k. Box plots: Key numerical features vs delay ---
print("\n--- 4k. Box Plots: Numerical Features vs Delay ---")
box_features = ['distance_km', 'scheduled_travel_hours', 'maintenance_score', 
                'coach_age_years', 'route_historical_ontime_pct', 'seat_utilisation_pct']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(box_features):
    df.boxplot(column=col, by='is_delayed', ax=axes[i])
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Is Delayed')
plt.suptitle('Key Features Distribution by Delay Status', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('plots/11_boxplots_vs_delay.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/11_boxplots_vs_delay.png")

# --- 4l. Delay trends over years ---
print("\n--- 4l. Delay Rate Trend Over Years ---")
delay_by_year = df.groupby('year')['is_delayed'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(delay_by_year.index, delay_by_year.values, 'bo-', markersize=10, linewidth=2)
ax.fill_between(delay_by_year.index, delay_by_year.values, alpha=0.2)
ax.set_title('Train Delay Rate Over Years (2018-2024)', fontsize=14, fontweight='bold')
ax.set_xlabel('Year')
ax.set_ylabel('Delay Rate')
ax.set_xticks(delay_by_year.index)
for x, y in zip(delay_by_year.index, delay_by_year.values):
    ax.annotate(f'{y:.1%}', (x, y), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('plots/12_delay_trend_years.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Saved: plots/12_delay_trend_years.png")


# ============================================================================
# 5. PREPROCESSING PIPELINE
# ============================================================================
print("\n" + "=" * 70)
print("  5. PREPROCESSING PIPELINE")
print("=" * 70)

# Step 1: Drop leakage & metadata columns
print("\n--- Step 1: Drop leakage and metadata columns ---")
drop_cols = ['journey_id', 'train_number', 'departure_date', 
             'primary_delay_cause', 'delay_minutes']
df_clean = df.drop(columns=drop_cols)
print(f"Dropped: {drop_cols}")
print(f"Remaining columns: {df_clean.shape[1]}")

# Step 2: Handle redundant columns
print("\n--- Step 2: Drop redundant columns ---")
# zone_abbr is redundant with zone; is_electrified is derivable from traction_type
redundant = ['zone_abbr']
df_clean = df_clean.drop(columns=redundant)
print(f"Dropped redundant: {redundant}")

# Step 3: Identify feature types
print("\n--- Step 3: Identify feature types ---")
categorical_cols = ['train_type', 'season', 'zone', 'source_station_category',
                    'destination_station_category', 'traction_type']
binary_cols = ['is_weekend', 'is_night_departure', 'is_peak_hour', 
               'is_festival_season', 'track_doubled', 'is_hdn_route',
               'is_electrified', 'is_circular_route', 'is_monsoon_season',
               'is_fog_risk', 'has_lhb_coaches', 'is_rake_shared',
               'is_overloaded', 'late_incoming_rake', 'is_special_train']
numerical_cols = ['year', 'month', 'day_of_week', 'departure_hour',
                  'distance_km', 'num_scheduled_stops', 'scheduled_travel_hours',
                  'psr_count', 'fog_risk_score', 'zone_fog_index',
                  'zone_congestion_index', 'season_severity_score',
                  'loco_age_years', 'coach_age_years', 'maintenance_score',
                  'seat_utilisation_pct', 'route_historical_ontime_pct']
target_col = 'is_delayed'

print(f"Categorical : {len(categorical_cols)} features")
print(f"Binary      : {len(binary_cols)} features")
print(f"Numerical   : {len(numerical_cols)} features")
print(f"Target      : {target_col}")

# Step 4: Encode categorical features
print("\n--- Step 4: Label Encode categorical features ---")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le
    print(f"  {col}: {len(le.classes_)} classes → encoded to 0-{len(le.classes_)-1}")

# Step 5: Separate features and target
print("\n--- Step 5: Separate X (features) and y (target) ---")
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y distribution:\n{y.value_counts()}")

# Step 6: Train-Test split
print("\n--- Step 6: Train-Test Split (80/20) ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts()}")
print(f"y_test distribution:\n{y_test.value_counts()}")

# Step 7: Feature Scaling (for distance-based models)
print("\n--- Step 7: Standard Scaling (numerical features) ---")
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
print(f"Scaled {len(numerical_cols)} numerical features using StandardScaler")
print(f"X_train_scaled shape: {X_train_scaled.shape}")

# Step 8: Save preprocessed data
print("\n--- Step 8: Save preprocessed data ---")
X_train_scaled.to_csv('data/X_train.csv', index=False)
X_test_scaled.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
print("[OK] Saved: data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv")

# Summary of preprocessing
print("\n--- Preprocessing Pipeline Summary ---")
print(f"""
╔══════════════════════════════════════════════════════════╗
║           PREPROCESSING PIPELINE SUMMARY                 ║
╠══════════════════════════════════════════════════════════╣
║ Step 1: Dropped leakage columns (5 cols)                 ║
║ Step 2: Dropped redundant columns (1 col)                ║
║ Step 3: Identified feature types                         ║
║ Step 4: Label Encoded 6 categorical features             ║
║ Step 5: Separated X ({X.shape[1]} features) and y (target)         ║
║ Step 6: Train-Test split (80/20, stratified)             ║
║ Step 7: StandardScaler on {len(numerical_cols)} numerical features       ║
║ Step 8: Saved preprocessed CSVs                          ║
╠══════════════════════════════════════════════════════════╣
║ Final X_train: {X_train_scaled.shape}                         ║
║ Final X_test : {X_test_scaled.shape}                          ║
╚══════════════════════════════════════════════════════════╝
""")


# ============================================================================
# 6. MODEL SHORTLISTING
# ============================================================================
print("\n" + "=" * 70)
print("  6. MODEL SHORTLISTING (6+ models, 2+ ensembles)")
print("=" * 70)

print("""
╔════╦═══════════════════════════════╦═══════════════╦════════════════════════════════════╗
║ #  ║ Model                         ║ Type          ║ Why Selected                       ║
╠════╬═══════════════════════════════╬═══════════════╬════════════════════════════════════╣
║  1 ║ Logistic Regression           ║ Linear        ║ Baseline, interpretable, fast       ║
║  2 ║ Decision Tree                 ║ Tree-based    ║ Interpretable, handles non-linear   ║
║  3 ║ K-Nearest Neighbors (KNN)     ║ Instance      ║ Non-parametric, simple              ║
║  4 ║ Support Vector Machine (SVM)  ║ Kernel-based  ║ Effective in high dimensions        ║
║  5 ║ Random Forest *               ║ ENSEMBLE (Bag)║ Robust, reduces overfitting         ║
║  6 ║ Gradient Boosting *           ║ ENSEMBLE (Bst)║ High accuracy, sequential learning  ║
║  7 ║ XGBoost *                     ║ ENSEMBLE (Bst)║ State-of-the-art, regularization    ║
║  8 ║ LightGBM *                    ║ ENSEMBLE (Bst)║ Fast training, handles large data   ║
╚════╩═══════════════════════════════╩═══════════════╩════════════════════════════════════╝

* = Ensemble Model (4 out of 8 are ensembles, meeting the 2+ requirement)

Ensemble Methods Used:
  • Bagging  : Random Forest (parallel ensemble of decision trees)
  • Boosting : Gradient Boosting, XGBoost, LightGBM (sequential error correction)
  
Primary Evaluation Metric: AUC-ROC Score
Secondary Metrics: Accuracy, Precision, Recall, F1-Score
""")


# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("  PHASE 1 COMPLETE [OK]")
print("=" * 70)

print(f"""
SUMMARY:
--------
[OK] 1. Dataset    : Indian Railways Predict Train Delay (1.5M records, 45 cols)
[OK] 2. Metadata   : 42 features across 6 groups, data dictionary included
[OK] 3. Problem    : Binary Classification (is_delayed: >15 min late?)
[OK] 4. EDA        : 12 visualizations saved in plots/ folder
[OK] 5. Pipeline   : Label Encoding + StandardScaler + 80/20 split
[OK] 6. Models     : 8 shortlisted (4 ensembles: RF, GB, XGBoost, LightGBM)

OUTPUT FILES:
  plots/01_target_distribution.png
  plots/02_numerical_distributions.png
  plots/03_delay_by_train_type.png
  plots/04_delay_by_season.png
  plots/05_delay_by_hour.png
  plots/06_delay_by_zone.png
  plots/07_correlation_heatmap.png
  plots/08_top_correlations.png
  plots/09_delay_by_station_category.png
  plots/10_delay_causes.png
  plots/11_boxplots_vs_delay.png
  plots/12_delay_trend_years.png
  data/X_train.csv
  data/X_test.csv
  data/y_train.csv
  data/y_test.csv
""")

print(">> Ready for Phase 2: Model Training & Evaluation!")
