# 🚆 Predicting Train Delays Using Machine Learning Classifiers and Boosting Models

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)

A Machine Learning Mini Project designed to predict whether a passenger train will experience an arrival delay of **more than 15 minutes** before departure, leveraging operational schedules, seasonal severity, track congestion, and route reliability factors.

---

## 📌 Project Highlights
- **Binary Classification:** Target `is_delayed` (1 = Delay > 15 mins, 0 = On-Time / Delay ≤ 15 mins).
- **Leak-Free Preprocessing:** Strictly excluded post-departure features (`delay_minutes`, `primary_delay_cause`) to ensure real-world validity.
- **6 Diverse Models Benchmarked:**
  - 4 Individual Baselines: Logistic Regression, Decision Tree, Gaussian Naive Bayes, K-Nearest Neighbors (KNN).
  - 2 Boosting Ensembles: **AdaBoost** and **Gradient Boosting** (operational champion).
- **Dual Deployment Options:**
  1. 🎈 **Streamlit Interactive Dashboard** (`streamlit_app.py`) with live prediction sliders, probability gauges, and embedded EDA plots.
  2. 🌐 **Flask Web Application** (`app.py`) with responsive UI and a REST API (`/api/predict`).

---

## 📊 Model Performance Comparison

All models were evaluated on a stratified test partition:

| Model Name | Architectural Type | Accuracy | Precision | Recall | F1-Score | **ROC-AUC** | Training Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **Ensemble (Boosting)** | **85.20%** | 87.57% | **92.65%** | **0.9004** | **0.9122** | 11.49s |
| **Logistic Regression** | Individual (Linear) | **85.65%** | **88.51%** | 0.9207 | **0.9026** | **0.9154** | 0.06s |
| **AdaBoost** | **Ensemble (Boosting)** | 84.66% | 87.85% | 0.9139 | 0.8958 | 0.9084 | 3.64s |
| **Decision Tree** | Individual (Tree) | 81.82% | 85.85% | 0.8958 | 0.8767 | 0.8563 | 0.38s |
| **K-Nearest Neighbors** | Individual (Instance) | 80.52% | 85.33% | 0.8817 | 0.8673 | 0.8404 | 0.01s |
| **Gaussian Naive Bayes** | Individual (Probabilistic) | 73.72% | **92.28%** | 0.6940 | 0.7922 | 0.8654 | 0.03s |

> **Operational Champion:** Gradient Boosting Classifier was selected for production because it achieves the highest **Recall (92.65%)**, minimizing missed delays.

---

## 🚀 How to Run Locally

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/shraddha85912004/train-delay-prediction.git
cd train-delay-prediction
pip install -r requirements.txt
```

### 2. Option A: Launch Streamlit Web Dashboard
```bash
streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.  
*(Or double-click `run_streamlit.bat` on Windows)*

### 3. Option B: Launch Flask Web Application & REST API
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.  
*(Or double-click `run_app.bat` on Windows)*

#### Testing the REST API:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 600, "late_incoming_rake": 1, "route_historical_ontime_pct": 55.0}'
```

---

## 📁 Repository Structure
```
train-delay-prediction/
├── app.py                              # Flask backend application
├── streamlit_app.py                    # Streamlit interactive application
├── train_delay_prediction.py           # Clean model training & evaluation script
├── Train_Delay_Prediction_Notebook.ipynb # Jupyter Notebook with pre-rendered outputs
├── PROJECT_REPORT.md                   # Complete academic project report
├── AI_Prompt_and_Command_Logs.md       # Full prompt & command audit logs
├── requirements.txt                    # Project dependencies
├── run_app.bat                         # Flask launcher script
├── run_streamlit.bat                   # Streamlit launcher script
├── saved_model/                        # Serialized model artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── metadata.json
├── templates/
│   └── index.html                      # Flask HTML template
├── static/
│   └── style.css                       # Web styles
├── plots/                              # Generated EDA & evaluation charts
└── data/
    └── ir_data_dictionary.csv          # Column descriptions
```

---

## 📜 Dataset Note
Due to GitHub file size limits, the raw dataset (`ir_train.csv`, 337 MB) is excluded via `.gitignore`. You can obtain the raw dataset from the [Kaggle Competition: Indian Railways Predict Train Delay](https://www.kaggle.com/competitions/133990).
