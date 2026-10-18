** NSW Transport Safety Analytics Pipeline **

## Project Overview
This repository contains the data engineering and exploratory data analysis (EDA) pipeline for processing historic New South Wales crash incident logs (2020–2024) and integrating regional demographic metrics. The output generates clean data arrays optimized for downstream binary classification and unsupervised clustering models.

## Repository Architecture
* `data_processing.ipynb`: Notebook executing data cleaning, regional NSW filtering, value melting, and the primary structural left-merge.
* `eda.ipynb`: Notebook running exploratory data analysis, bar plotting, and categorical relationship testing using Chi-Square and Cramér's V metrics.
* `processed/`: Directory hosting the final cleaned datasets (`nsw_crashes_classification.csv` and `lga_yearly_aggregates_clustering.csv`).
* `figures/`: Folder containing all 6 exported high-fidelity performance charts.

## Setup & Execution Guide

### 1. Environment Requirements
Ensure your Python environment contains the following core packages:

"pip install pandas numpy matplotlib scipy scikit-learn joblib"

### 2. Execution Pipeline Sequence
To completely reproduce the pipeline findings, execute the notebooks in this order:
1. Run all cells in `data_processing.ipynb` to read raw data inputs, strip strings, match LGA names, and populate the `processed/` data folder.
2. Run all cells in `eda.ipynb` to ingest the processed files, perform statistical tests, and export the horizontal/vertical graphics to `figures/`.

### 3. Visualisation Charts
Each script reads its input files from the folder we run it in, so copy the input files next to the script first, then run the command from that folder.

*'Clustering Charts': Needs `lga_clusters.csv` and `kmeans_k_scores.csv
To run we use python cluster_charts.py 
It creates `chart_cluster_scatter.png`, `chart_cluster_profile.png` and `chart_cluster_silhouette.png`.

*'Classification charts': Needs `logreg_test_predictions.csv` and `logreg_permutation_importance.csv`
To run we use python classification_charts.py 
It creates `chart_confusion_matrix.png`, `chart_class_metrics.png` and `chart_feature_importance.png`.

## 4. Clustering Model (K-Means and DBSCAN)

### Purpose
The clustering notebook answers two questions:
1. Which NSW LGAs have similar road crash patterns? (K-Means)
2. Which LGA-year records are unusual and fit no group? (DBSCAN)

### Input
Used `processed/lga_yearly_aggregates_clustering.csv` which as four features Crash rate per 100k, Severe % (the share of crashes with a casualty), Wet-weather % (the share of crashes on a wet road surface) and Rural-road %.

### How to run it
1. Run `data_processing.ipynb` first, so the input file exists in `processed/`.
2. Open `clustering-data.ipynb` and run all cells from top to bottom.The results are saved to `clusteringcsv/`, and the K-Means model is saved to `models/kmeans_lga.joblib`.

### Settings
random_state = 20, k = 3, eps = 1.0, min_samples = 5.

### How to read the results
* Cluster 0: rural, highest casualty share, lowest wet-surface share.
* Cluster 1: mostly non-rural, lowest crash rate.
* Cluster 2: rural, highest wet-surface share, highest crash rate.

### Running a saved model 
This is only used for k-means prediction
`predict_cluster.py` assigns new records to a K-Means cluster using the saved model. Run it from the folder that contains `models/`: 

python predict_cluster.py processed/lga_yearly_aggregates_clustering.csv

The script then will be saved in `cluster_predictions.csv`. 

## 5. Classificatin Model (Random Forest(crash casualty.ipynb) and Logistic regression(logistic regression.ipynb))

### Purpose
The cLassification models tries to answers these questions respectiviely:
1. Can environmental and road-condition features predict whether an individual crash will be a casualty crash? (crash casualty.ipynb)
2. Can an LGA's past crash profile based on crash rate, weather and road condition predict whether it will be a high-risk LGA in the following year? (logistic regression.ipynb)

### Input
## Crash casualty.ipynb
Uses processed/nsw_crashes_classification.csv: one row per crash, 90,857 crashes from 2020 to 2024.
Target Severe_Crash: 1 = casualty crash (someone was killed or injured), 0 = non-casualty towaway. 67.93% of crashes are 1, so always predicting 1 scores 67.93% accuracy. This is the baseline to beat.
Features: Weather, Road classification (admin), Road surface, Surface condition, Urbanisation, Speed limit, Season and Population.
Crash ID and LGA code are dropped because they are identifiers.
## logistic regression.ipynb
Uses processed/lga_yearly_aggregates_clustering.csv: 645 rows (129 LGAs x 5 years).
Features: Population, Severe %, Wet-weather %, Rural-road %.

### How to run it
1.Run data_processing.ipynb first, so both input files exist in processed/.
2. Open crash casualty.ipynb and logistic regression.ipynb and run all cells from top to bottom in each. The results are saved to processed/.
### Settings
Both: random_state = 20.
## Crash casualty.ipynb
Stratified 80/20 train/test split (72,685 training crashes, 18,172 test crashes).
One-hot encoding inside the pipeline, with Speed limit treated as a category and Population standardised.
class_weight = balanced for both models.
Logistic Regression: C = 1. Random Forest: 150 trees, max_depth = 10, min_samples_leaf = 20. Both were chosen by grid search on the training data only.
Decision threshold on P(non-casualty), chosen on training data by balanced accuracy: 0.51 (Logistic Regression) and 0.50 (Random Forest).
## logistic regression.ipynb
Time-based split: train on 2020 to 2023 (516 records), test on 2024 (129 records).
StandardScaler + LogisticRegression (max_iter = 1000).
5-fold GroupKFold cross-validation, grouped by LGA.

### How to read the results
## Crash casualty.ipynb
results in test evaluation. Metric provided: Accuracy, Precision (NS), Recall (NS), F1 (NS), Balanced account, ROC-AUC,and PR-AUC (NS)
## logistic regression.ipynb
results provided in test evaluation Metric provided: Accuract, Precision, Recall, F1, dummy accuracy, Test ROC-AUC,and CV ROC-AUC
