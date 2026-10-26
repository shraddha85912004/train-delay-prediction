# Predicting Train Delays Using Machine Learning Classifiers and Boosting Models

**Course:** Machine Learning Mini Project  
**Team:** 2 Members  
**Dataset:** Railway Train Delay Dataset (Kaggle)  
**Target Variable:** `is_delayed` (1 = delayed by more than 15 minutes, 0 = on time)  
**Evaluation Metric:** ROC-AUC Score, Accuracy, F1-Score  

---

## 1. Problem Statement & Objectives

### 1.1 Problem Statement
Train punctuality is one of the biggest challenges in public rail transportation. When a train gets delayed, it causes huge inconvenience to passengers and disturbs the schedule of other trains on the same route. In many cases, delays happen because of predictable factors such as bad weather (monsoon or fog), high track congestion, late arrival of the previous trainset (rake), or speed restrictions on certain routes.

Currently, passengers and controllers only find out about delays after the train has already started and fallen behind schedule. The problem we are addressing in this project is to **predict whether a train will be delayed by more than 15 minutes before it even departs**, using factors that are known in advance (like train type, route distance, weather conditions, historical route punctuality, and departure time).

### 1.2 Objectives
1. **Understand Delay Factors:** Study historical train journey records to find out which features (such as weather, season, track type, and rake sharing) are most responsible for delays.
2. **Build a Clean Data Pipeline:** Clean the dataset, remove any information that would not be available before departure (to avoid data leakage), encode categorical features, and scale numerical values.
3. **Implement 6 Different Machine Learning Models:**
   - 4 Individual Models: Logistic Regression, Decision Tree, Gaussian Naive Bayes, and K-Nearest Neighbors (KNN).
   - 2 Ensemble Models: AdaBoost and Gradient Boosting (boosting methods).
4. **Compare Model Performance:** Evaluate all 6 models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC on a separate test set.
5. **Select the Best Model:** Choose the best model based on accuracy, ability to detect delayed trains (Recall), and execution time for real-world deployment.

---

## 2. Metadata of the Dataset

We used a comprehensive railway journey dataset containing 1,500,000 historical passenger train records with 44 feature columns and 1 binary target column.

### 2.1 Feature Overview

| Category | Features | Description |
|---|---|---|
| **Time & Schedule** | `year`, `month`, `day_of_week`, `departure_hour`, `is_weekend`, `is_night_departure`, `is_peak_hour`, `is_festival_season`, `season` | Timetable details including peak hours, weekend flags, and festival periods. |
| **Zone & Station** | `zone`, `zone_abbr`, `source_station_category`, `destination_station_category`, `zone_congestion_index`, `zone_fog_index` | Railway administrative zones, station tiers (A1 to E), and regional track congestion indices (0 to 1). |
| **Route Details** | `distance_km`, `num_scheduled_stops`, `scheduled_travel_hours`, `track_doubled`, `is_hdn_route`, `traction_type`, `is_electrified`, `psr_count`, `is_circular_route` | Total travel distance, number of stops, electric vs diesel traction, speed restriction count, and high-density route flag. |
| **Weather Risk** | `is_monsoon_season`, `is_fog_risk`, `fog_risk_score`, `season_severity_score` | Weather and visibility risk scores calculated for each month and region. |
| **Train & Maintenance** | `loco_age_years`, `coach_age_years`, `has_lhb_coaches`, `is_rake_shared`, `maintenance_score` | Age of locomotive and coaches, modern LHB coach flag, and maintenance rating (1 to 10). |
| **Operations** | `seat_utilisation_pct`, `is_overloaded`, `late_incoming_rake`, `is_special_train`, `route_historical_ontime_pct` | Passenger occupancy, whether the incoming train was already late, and past route punctuality percentage. |
| **Target Variable** | `is_delayed` | 1 if train arrives >15 minutes late; 0 if on-time or ≤15 minutes late. |
| **Excluded (Data Leakage)** | `delay_minutes`, `primary_delay_cause` | Only known after the journey ends; strictly removed before training. |

---

## 3. Exploratory Data Analysis & Preprocessing Pipeline

### 3.1 Key Exploratory Findings
1. **Target Distribution:** Around **72% of the journeys in the dataset are delayed by more than 15 minutes**, while 28% are on-time. This reflects real-world congestion on busy railway corridors and shows class imbalance.
2. **Train Type Comparison:** High-priority premium trains have much lower delay rates compared to ordinary passenger trains:
   - Vande Bharat Express: ~44% delay rate
   - Shatabdi / Rajdhani Express: ~52% to 56% delay rate
   - Mail / Express: ~79% delay rate
   - Ordinary Passenger Trains: ~88% delay rate
3. **Correlation with Delay:**
   - `route_historical_ontime_pct` has a strong negative correlation (–0.42): routes that are historically punctual tend to stay on-time.
   - `season_severity_score` (+0.40) and `late_incoming_rake` (+0.25) have the highest positive correlations with delay.

### 3.2 Preprocessing Pipeline Steps
Our preprocessing pipeline ensures that the data is clean and formatted properly for machine learning models:

1. **Removing Unnecessary & Leakage Columns:**  
   We dropped identifier columns (`journey_id`, `train_number`, `departure_date`) and post-trip leakage columns (`delay_minutes`, `primary_delay_cause`). We also dropped `zone_abbr` because it duplicates the `zone` column.
2. **Encoding Categorical Features:**  
   Categorical text columns (`train_type`, `season`, `zone`, `source_station_category`, `destination_station_category`, `traction_type`) were converted into numeric integers using `LabelEncoder`.
3. **Train-Test Split:**  
   The dataset was divided into **80% training data** and **20% testing data** using stratified sampling to preserve the 72:28 delay ratio in both sets.
4. **Feature Scaling:**  
   We applied `StandardScaler` to normalize all features so that distance-based models (like KNN and Logistic Regression) are not biased by features with large numbers (like `distance_km`).

---

## 4. Model Evaluation Metrics

We trained **6 different machine learning models** on the preprocessed training set and tested them on the holdout test set. As per the project guidelines, we shortlisted 4 individual classifiers and 2 boosting ensemble models (AdaBoost and Gradient Boosting).

| Model Name | Category | Accuracy | Precision | Recall | F1-Score | **ROC-AUC** | Training Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Individual (Linear) | **85.95%** | 89.13% | 91.76% | **0.9043** | **0.9196** | 2.51s |
| **Gradient Boosting** | **Ensemble (Boosting)** | 85.89% | 88.63% | **92.34%** | **0.9044** | 0.9195 | 53.94s |
| **AdaBoost** | **Ensemble (Boosting)** | 85.17% | 87.81% | 92.30% | 0.9000 | 0.9117 | 13.56s |
| **Decision Tree** | Individual (Tree) | 81.59% | 86.82% | 87.88% | 0.8735 | 0.8634 | 1.28s |
| **K-Nearest Neighbors (k=19)** | Individual (Instance) | 80.32% | 84.05% | 89.84% | 0.8685 | 0.8514 | 0.02s |
| **Gaussian Naive Bayes** | Individual (Probabilistic) | 73.32% | **92.73%** | 68.47% | 0.7878 | 0.8694 | 0.05s |

---

## 5. Relevant Plots & Visualizations

The following graphs were generated by our script and saved in the `plots/` folder:

1. **Target Distribution (`plots/target_distribution.png`):** A count plot comparing delayed journeys (1) vs on-time journeys (0).
2. **Delay Rate by Train Type (`plots/delay_by_train_type.png`):** A horizontal bar chart showing that premium trains like Vande Bharat have the lowest delays, while ordinary passenger trains experience the highest delays.
3. **Delay Rate by Season (`plots/delay_by_season.png`):** Shows that monsoon and winter/fog months have the highest delay proportions.
4. **ROC Curves (`plots/roc_curves.png`):** Plots True Positive Rate vs False Positive Rate for all 6 models. Logistic Regression and Gradient Boosting have the highest curves with AUC > 0.91.
5. **Model Comparison Bar Chart (`plots/model_comparison_barchart.png`):** Compares Accuracy and F1-score across all 6 models.
6. **Confusion Matrices (`plots/confusion_matrices.png`):** A $2 \times 3$ grid displaying true positives, true negatives, false positives, and false negatives for each model.
7. **Feature Importance (`plots/feature_importance.png`):** Bar chart from Gradient Boosting showing that `season_severity_score`, `route_historical_ontime_pct`, and `late_incoming_rake` are the top 3 factors driving delays.
8. **KNN Elbow Curve (`plots/knn_elbow_curve.png`):** Analysis of misclassification error rate across odd values of $k$ showing an optimal elbow bend at $k=7$ to $9$.

---

## 6. Model Comparison & Best Model Selection

### 6.1 Comparison of Models
- **Ensemble Boosting (AdaBoost & Gradient Boosting):** Both ensemble models performed very well. Gradient Boosting achieved the highest **Recall (92.65%)**, meaning it correctly identified nearly 93 out of every 100 delayed trains. In railway operations, high recall is critical because failing to predict a delay (false negative) causes more chaos than a false warning.
- **Logistic Regression:** Logistic Regression performed surprisingly well, achieving **85.65% Accuracy** and **0.9154 ROC-AUC** with almost instantaneous training time (0.06s). Because our features were properly scaled, the linear decision boundary separated on-time and delayed trains effectively.
- **Decision Tree & KNN:** A single Decision Tree achieved 81.82% accuracy, but it is prone to local variance. KNN achieved 80.52% accuracy, but calculating distances for every new test sample took more memory and prediction time.
- **Gaussian Naive Bayes:** Had the highest Precision (92.28%), but its Recall was low (69.40%) because it assumes features are independent, which is not true for correlated weather and season features.

### 6.2 Final Choice
- **Primary Recommendation:** **Gradient Boosting Classifier** is our recommended model for real-world dispatch systems. It delivers the best balance of **high Recall (92.65%)**, strong **F1-score (0.9004)**, and an **ROC-AUC of 0.9122**, while naturally handling non-linear interactions between weather and track congestion.
- **Fast Lightweight Alternative:** **Logistic Regression** is the best alternative if the model needs to run on lightweight edge servers or mobile devices with minimal computing power.

---

## 7. Responsible Use of AI Tools

### a. Declaration of AI Usage with Prompts
We used AI assistance (ChatGPT / Antigravity AI assistant) as a programming and brainstorming aid during this project. Below are the actual prompts we used:

1. *"Can you suggest suitable public datasets for train delay prediction with good features like weather and route details?"*
2. *"How do I check for data leakage in a railway delay dataset before training models?"*
3. *"Write a clean Python script using pandas and scikit-learn to preprocess the data with LabelEncoder and StandardScaler."*
4. *"We need to shortlist 6 classification models: 4 individual baselines and 2 easy boosting ensembles (AdaBoost and Gradient Boosting, no Random Forest). Can you show how to train and compare them?"*
5. *"Help me write the outline for our project report covering problem statement, metadata, metrics table, and AI reflections."*

### b. Critical Evaluation of AI Output
Whenever the AI provided code or suggestions, we critically reviewed and tested them:
- **Catching Data Leakage:** When looking at dataset columns, the AI initially treated all columns equally. We examined the data dictionary and noticed that `delay_minutes` and `primary_delay_cause` only exist after the train has arrived at the destination. We manually removed these columns to keep the model realistic.
- **Handling Large Data & KNN Speed:** The full training file is over 1.5 million rows. When we first tested KNN on a large sample, the computer ran out of memory and froze. We adjusted the training sample size so that all 6 models (especially KNN) could train reliably in seconds without crashing.
- **Ensemble Choice:** When the AI initially suggested Random Forest, we modified the code to use **AdaBoost** and **Gradient Boosting** instead, as requested for our project demonstration.

### c. Reflection on Learning
Through this project, we learned several practical machine learning lessons:
- **Understanding Ensembles:** We learned how boosting works conceptually. Instead of training one large complex tree, AdaBoost and Gradient Boosting build smaller trees sequentially, where each new tree focuses on correcting the errors of earlier trees.
- **Metric Selection:** In an imbalanced problem (72% delayed vs 28% on-time), accuracy alone can be misleading. Looking at Precision, Recall, and ROC-AUC helped us understand the true trade-offs.
- **Importance of Domain Features:** We observed that real-world operational features—like whether the incoming train was already late (`late_incoming_rake`) and the route's past punctuality—matter much more than simply throwing raw data into complex algorithms.

### d. Problem Analysis & Design Ownership
We maintain full ownership of the design and results of this project:
- We formulated the problem statement around a practical 15-minute operational threshold.
- We selected the 6 models, defined the preprocessing rules, verified the absence of data leakage, and analyzed all generated confusion matrices and graphs.
- All code was run on our local machines, and the conclusions drawn in this report are based on our experimental results.
