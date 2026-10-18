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

## Settings
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
