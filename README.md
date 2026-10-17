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

"pip install pandas numpy matplotlib scipy"

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


