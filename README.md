# NSW Transport Safety Analytics Pipeline

## Project Overview
This repository contains the complete end-to-end machine learning pipeline for processing historic New South Wales crash incident logs (2020–2024). It integrates regional demographic metrics to generate clean datasets optimized for binary classification (crash severity prediction) and unsupervised clustering (LGA risk profiling). The pipeline includes data processing, exploratory data analysis (EDA), model training, and interactive prediction scripts.

## Repository Architecture
*   `data/`: Contains raw data (`raw/`) and generated processed datasets (`processed/`).
*   `models/`: Contains the saved `.joblib` models ready for deployment.
*   `figures/`: Contains all exported high-fidelity performance charts.
*   `notebooks/`: Contains the Jupyter notebooks for data processing, EDA, and model training.
*   `predict.py`: Script to predict crash severity for individual crashes.
*   `predict_cluster.py`: Script to assign new LGA records to a K-Means cluster.
*   `predict_lga_risk.py`: Script to predict if an LGA is high-risk.
*   `classification_charts.py`: Script to generate visualisations for the classification model.
*   `cluster_charts.py`: Script to generate visualisations for the clustering model.

---

## 1. Setup & Execution Guide

### Environment Requirements
Ensure your Python environment (Python 3.9+) contains the following core packages. You can install them using:
pip install pandas numpy matplotlib scipy scikit-learn joblib

### Execution Pipeline Sequence
To completely reproduce the pipeline findings, execute the notebooks inside the notebooks/ folder in this order:

1. Run data-processing.ipynb to read raw data inputs, clean LGA names, merge population data, and populate the data/processed/ folder.

2. Run eda.ipynb to ingest the processed files, perform statistical tests (Chi-Square, Cramér's V), and export charts to the figures/ folder.

3. Run crash_casualty.ipynb to train the Random Forest classification model and save the .joblib file to the models/ folder.

4. Run logistic regression.ipynb to train the Logistic Regression model and save the .joblib file to the models/ folder.

5. Run clustering-data.ipynb to train the K-Means and DBSCAN models, and save the cluster model to models/.

### Visualisation Charts
To recreate the model result charts, run these scripts from the root folder. They will automatically save the generated charts into the figures/ directory.

Classification Charts: python3 classification_charts.py (Creates confusion matrix, class metrics, probability distribution, ROC/PR curves, and feature importance charts).

Clustering Charts: python3 cluster_charts.py (Creates a PCA scatter plot, cluster profile bar chart, and silhouette score line chart).

## 2. Prediction Scripts (Testing the Models)

To test the trained models, use the provided command-line scripts. Each script requires a .csv input file containing the necessary features. Run these commands from the root Assignment 2 folder.

### A. Crash Severity Prediction (predict.py)
Predicts whether a crash will be a casualty or non-casualty based on environmental and road features.

Command: python3 predict.py data/processed/nsw_crashes_classification.csv

Output: Saves predictions.csv to the data/processed/ folder.

Required Columns: Weather, Road classification (admin), Road surface, Surface condition, Urbanisation, Season, Speed limit, Population.

### B. LGA Cluster Assignment (predict_cluster.py)
Assigns LGA-year records to one of the 3 K-Means clusters based on crash patterns.

Command: python3 predict_cluster.py data/processed/lga_yearly_aggregates_clustering.csv

Output: Saves cluster_predictions.csv to the data/processed/ folder.

Required Columns: Crash rate per 100k, Severe %, Wet-weather %, Rural-road %.

### C. LGA High-Risk Prediction (predict_lga_risk.py)
Predicts if an LGA will be high-risk based on its historical crash profile.

Command: python3 predict_lga_risk.py data/processed/lga_yearly_aggregates_clustering.csv

Output: Saves lga_risk_predictions.csv to the data/processed/ folder.

Required Columns: Population, Severe %, Wet-weather %, Rural-road %.



## 3. Model Details
### Clustering Models (K-Means and DBSCAN)

Purpose: Answers which NSW LGAs have similar road crash patterns (K-Means) and which LGA-year records are unusual (DBSCAN).

Inputs: data/processed/lga_yearly_aggregates_clustering.csv.

Features: Crash rate per 100k, Severe %, Wet-weather %, Rural-road %.

Settings: random_state = 20, k = 3, eps = 1.0, min_samples = 5.

Cluster Profiles:

Cluster 0: Rural, highest casualty share, lowest wet-surface share.

Cluster 1: Mostly non-rural, lowest crash rate.

Cluster 2: Rural, highest wet-surface share, highest crash rate.

### Classification Models (Random Forest and Logistic Regression)

Purpose: Answers if environmental features predict an individual crash casualty (Random Forest) and if an LGA's past profile predicts future high-risk status (Logistic Regression).

#### Crash Severity (crash_casualty.ipynb):

Input: data/processed/nsw_crashes_classification.csv.

Target: Severe_Crash (1 = casualty, 0 = non-casualty). Baseline accuracy is 67.93%.

Settings: Stratified 80/20 split. class_weight = balanced. Random Forest uses 150 trees, max_depth = 10. Logistic Regression uses C = 1.

#### LGA Risk (logistic regression.ipynb):

Input: data/processed/lga_yearly_aggregates_clustering.csv.

Settings: Time-based split (train 2020-2023, test 2024). StandardScaler + Logistic Regression. 5-fold GroupKFold cross-validation grouped by LGA.

The cLassification models tries to answers these questions respectiviely:
1. Can environmental and road-condition features predict whether an individual crash will be a casualty crash? (crash casualty.ipynb)
2. Can an LGA's past crash profile based on crash rate, weather and road condition predict whether it will be a high-risk LGA in the following year? (logistic regression.ipynb)
