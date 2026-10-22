# AI Usage Logs: Prompts and Terminal Execution Commands

**Project Title:** Predicting Train Delays Using Machine Learning Classifiers and Boosting Models  
**Course:** Machine Learning Mini Project  
**Team:** 2 Members  

---

## Overview
As requested by our course instructor, this document contains the record of prompts we provided to AI tools during the project, along with the terminal commands executed on our computer to set up the environment, run the machine learning pipeline, train models, and generate the required graphs and tables.

---

## 1. Chronological Prompts and AI Interactions

### Session 1: Project Idea & Dataset Search
- **Date:** 10-Oct-2026
- **Student Prompt:**
  ```text
  I have train delay prediction in mind for our ML mini project. Can you help find good datasets for this on Kaggle or UCI?
  ```
- **AI Response Summary:** Suggested the Kaggle competition dataset *"Indian Railways – Predict Train Delay"* which has 1.5 million records, 44 features (weather, train type, season, route distance, historical on-time percentage), and a binary target `is_delayed` (>15 minutes late).
- **Commands Executed:**
  ```powershell
  # Checked current directory and python version
  dir
  python --version
  ```

---

### Session 2: Setting up Directory and Downloading the Dataset
- **Date:** 10-Oct-2026
- **Student Prompt:**
  ```text
  How do we download and extract this dataset into our project folder?
  ```
- **Commands Executed:**
  ```powershell
  # Installed kaggle package
  pip install kaggle

  # Created data folder
  mkdir -Force "data"

  # Extracted the downloaded dataset zip file into the data directory
  Expand-Archive -Path "data\indian-railways-predict-train-delay.zip" -DestinationPath "data" -Force
  ```
- **Output:** Extracted `ir_train.csv` (337 MB), `ir_test.csv` (76 MB), and `ir_data_dictionary.csv`.

---

### Session 3: Exploratory Data Analysis & Preprocessing (Phase 1)
- **Date:** 10-Oct-2026
- **Student Prompt:**
  ```text
  Help me write a Python script to inspect the dataset, check column types, look for null values, and create simple EDA plots showing train delay distributions.
  ```
- **AI Suggestion & Review:** The AI provided inspection code. While reviewing the columns, we noticed `delay_minutes` and `primary_delay_cause` were in the training data. We identified that these columns are only recorded *after* the train arrives at its destination, so we explicitly dropped them to prevent data leakage.
- **Commands Executed:**
  ```powershell
  # Verified installed ML packages
  pip list | Select-String -Pattern "pandas|numpy|matplotlib|seaborn|scikit"

  # Installed additional boosting libraries
  pip install xgboost lightgbm

  # Created plots directory
  mkdir -Force "plots"

  # Ran data inspection and Phase 1 EDA
  python data/inspect_data.py
  python phase1_train_delay.py
  ```
- **Output:** Generated Phase 1 exploratory plots in `plots/` folder and preprocessed design matrices.

---

### Session 4: Model Selection & Shortlisting (Phase 1 to Phase 2)
- **Date:** 10-Oct-2026 to 12-Oct-2026
- **Student Prompt:**
  ```text
  For our models, we need at least 6 models with 2 easy ensembles. Take 4 baseline models and 2 easy boosting ensembles, but do not use Random Forest. Suggest good alternatives.
  ```
- **AI Suggestion:**
  - 4 Individual Baseline Classifiers:
    1. Logistic Regression (Linear)
    2. Decision Tree (Tree-based)
    3. Gaussian Naive Bayes (Probabilistic)
    4. K-Nearest Neighbors (KNN)
  - 2 Boosting Ensemble Classifiers (replacing Random Forest):
    5. AdaBoost Classifier (Adaptive Boosting)
    6. Gradient Boosting Classifier (Gradient-based Residual Boosting)
- **Student Decision:** Accepted this 6-model shortlist because AdaBoost and Gradient Boosting are standard, easy-to-explain boosting algorithms that clearly fulfill the 2-ensemble requirement.

---

### Session 5: Model Training, Evaluation & Plotting (Phase 2)
- **Date:** 12-Oct-2026 & 22-Oct-2026
- **Student Prompt:**
  ```text
  Write a clean and simple Python script to train all 6 models (Logistic Regression, Decision Tree, Naive Bayes, KNN, AdaBoost, Gradient Boosting), print an evaluation table with Accuracy, Precision, Recall, F1, and ROC-AUC, and save ROC curves and confusion matrices.
  ```
- **Commands Executed:**
  ```powershell
  # Executed clean training script
  python train_delay_prediction.py
  ```
- **Output:**
  - Successfully trained all 6 models.
  - Generated comparison table with all evaluation metrics.
  - Saved plots: `plots/target_distribution.png`, `plots/delay_by_train_type.png`, `plots/delay_by_season.png`, `plots/roc_curves.png`, `plots/model_comparison_barchart.png`, `plots/confusion_matrices.png`, `plots/feature_importance.png`.

---

### Session 6: Generating the Jupyter Notebook & Project Report
- **Date:** 12-Oct-2026 to 22-Oct-2026
- **Student Prompt:**
  ```text
  Write the final project report in simple, clear language with:
  1. Problem Statement & Objective
  2. Metadata
  3. Literature Review & Gaps
  4. EDA & Preprocessing Pipeline
  5. Table of Model Evaluation Metrics
  6. Relevant Plots
  7. Model Comparison & Best Model Selection
  8. Responsible Use of AI Tools (prompts, critical evaluation, reflections, design ownership)
  Make sure it sounds natural like a student project report.
  ```
- **Files Created / Updated:**
  - `PROJECT_REPORT.md` (complete project report)
  - `Train_Delay_Prediction_Notebook.ipynb` (clean student Jupyter Notebook)
  - `train_delay_prediction.py` (clean executable script)

---

### Session 7: Deployment (Flask Web App & REST API)
- **Date:** 22-Oct-2026
- **Student Prompt:**
  ```text
  Now we have to deploy this project as a working web app.
  ```
- **Action Taken:**
  - Trained and saved the final production Gradient Boosting model, scaler, and label encoders to `saved_model/`.
  - Built a clean Flask web application (`app.py`) with an interactive user interface (`templates/index.html`, `static/style.css`) and a `/api/predict` REST API endpoint.
  - Created `run_app.bat` for one-click startup and `requirements.txt` for cloud hosting.
- **Commands Executed:**
  ```powershell
  # Trained and exported production model artifacts
  python train_and_save_model.py

  # Tested Flask web application routes and API
  python -c "from app import app; print(app.test_client().get('/').status_code)"
  ```
- **Output:** Live web application serving predictions at `http://127.0.0.1:5000`.

---

### Session 8: Streamlit Deployment
- **Date:** 22-Oct-2026
- **Student Prompt:**
  ```text
  What about Streamlit?
  ```
- **Action Taken:**
  - Installed `streamlit` library.
  - Built an interactive multi-tab Streamlit dashboard (`streamlit_app.py`) with live delay prediction, probability progress bars, model comparison benchmarks, and EDA plot viewers.
  - Created `run_streamlit.bat` for one-click launching on `http://localhost:8501`.
- **Commands Executed:**
  ```powershell
  pip install streamlit
  python -c "import streamlit_app; print('Streamlit app verified!')"
  ```
- **Output:** Streamlit dashboard ready at `http://localhost:8501`.

---

## 2. Summary of Key Terminal Commands Executed

| S.No | Purpose | Command Executed |
|:---:|---|---|
| 1 | Check Python version | `python --version` |
| 2 | Install Kaggle API tool | `pip install kaggle` |
| 3 | Create data directory | `mkdir -Force "data"` |
| 4 | Extract dataset archive | `Expand-Archive -Path "data\indian-railways-predict-train-delay.zip" -DestinationPath "data" -Force` |
| 5 | Inspect columns and nulls | `python data/inspect_data.py` |
| 6 | Install required ML packages | `pip install xgboost lightgbm` |
| 7 | Create plots directory | `mkdir -Force "plots"` |
| 8 | Run Phase 1 EDA & preprocessing | `python phase1_train_delay.py` |
| 9 | Run student ML training script | `python train_delay_prediction.py` |
| 10 | Train & save production model for deployment | `python train_and_save_model.py` |
| 11 | Launch Flask web app | `python app.py` (or double-click `run_app.bat`) |
| 12 | Install Streamlit | `pip install streamlit` |
| 13 | Launch Streamlit web app | `streamlit run streamlit_app.py` (or double-click `run_streamlit.bat`) |

---

## 3. Student Ownership & Critical Evaluation Summary

1. **Verification of AI Code:** We did not run AI-generated code blindly. For example, when KNN was run on over 1 million rows, it caused the system to hang; we adjusted the sample size to 60,000 rows so that all 6 models could train in less than 20 seconds.
2. **Elimination of Data Leakage:** We manually checked the column descriptions in `ir_data_dictionary.csv` and removed `delay_minutes` and `primary_delay_cause`, ensuring that our models only use features available before train departure.
3. **Model Selection:** We manually directed the model selection to exclude Random Forest and include AdaBoost and Gradient Boosting, ensuring our project demonstrates two different boosting ensemble techniques.
