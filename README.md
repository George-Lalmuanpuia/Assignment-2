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

The script then will be saved in `cluster_predictions.csv`. 

## 5. Classificatin Model (Random Forest(crash casualty.ipynb) and Logistic regression(logistic regression.ipynb))

### Purpose
The cLassification models tries to answers these questions respectiviely:
1. Can environmental and road-condition features predict whether an individual crash will be a casualty crash? (crash casualty.ipynb)
2. Can an LGA's past crash profile based on crash rate, weather and road condition predict whether it will be a high-risk LGA in the following year? (logistic regression.ipynb)

### Input
### Crash casualty.ipynb
Uses processed/nsw_crashes_classification.csv: one row per crash, 90,857 crashes from 2020 to 2024.
Target Severe_Crash: 1 = casualty crash (someone was killed or injured), 0 = non-casualty towaway. 67.93% of crashes are 1, so always predicting 1 scores 67.93% accuracy. This is the baseline to beat.
Features: Weather, Road classification (admin), Road surface, Surface condition, Urbanisation, Speed limit, Season and Population.
Crash ID and LGA code are dropped because they are identifiers.
### logistic regression.ipynb
Uses processed/lga_yearly_aggregates_clustering.csv: 645 rows (129 LGAs x 5 years).
Features: Population, Severe %, Wet-weather %, Rural-road %.

### How to run it
1.Run data_processing.ipynb first, so both input files exist in processed/.
2. Open crash casualty.ipynb and logistic regression.ipynb and run all cells from top to bottom in each. The results are saved to processed/.
### Settings
Both: random_state = 20.
### Crash casualty.ipynb
Stratified 80/20 train/test split (72,685 training crashes, 18,172 test crashes).
One-hot encoding inside the pipeline, with Speed limit treated as a category and Population standardised.
class_weight = balanced for both models.
Logistic Regression: C = 1. Random Forest: 150 trees, max_depth = 10, min_samples_leaf = 20. Both were chosen by grid search on the training data only.
Decision threshold on P(non-casualty), chosen on training data by balanced accuracy: 0.51 (Logistic Regression) and 0.50 (Random Forest).
### logistic regression.ipynb
Time-based split: train on 2020 to 2023 (516 records), test on 2024 (129 records).
StandardScaler + LogisticRegression (max_iter = 1000).
5-fold GroupKFold cross-validation, grouped by LGA.

### How to read the results
### Crash casualty.ipynb
results in test evaluation. Metric provided: Accuracy, Precision (NS), Recall (NS), F1 (NS), Balanced account, ROC-AUC,and PR-AUC (NS)
### logistic regression.ipynb
results provided in test evaluation Metric provided: Accuract, Precision, Recall, F1, dummy accuracy, Test ROC-AUC,and CV ROC-AUC
