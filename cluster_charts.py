"""
cluster_charts.py
Draws the clustering charts

Input files:
  lga_clusters.csv     - one row per LGA-year, with the K-Means label and DBSCAN label
  kmeans_k_scores.csv  - (optional) silhouette score for each k; if missing, the script recalculates it

Charts made:
  chart_cluster_scatter.png     - the 4 clustering features squashed to 2D with PCA, coloured by K-Means cluster
  chart_cluster_profile.png     - average of each feature in each K-Means cluster
  chart_cluster_silhouette.png  - silhouette score for each k

Run with: python cluster_charts.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CSV_FILE = "data/processed/lga_clusters.csv"
SCORES_FILE = "data/processed/kmeans_k_scores.csv"

LGA_COL = "LGA"
KMEANS_COL = "kmean data"
DBSCAN_COL = "dbscans data"

FEATURES = ["Crash rate per 100k", "Severe %", "Wet-weather %", "Rural-road %"]

# Names used to describe each cluster in the scatter chart
CLUSTER_NAMES = {0: "Rural, highest severe %",
                 1: "Mostly non-rural, low crash rate",
                 2: "Rural, wettest roads, highest crash rate"}

def cname(cluster):
    """Get the cluster number and its name."""
    if cluster in CLUSTER_NAMES:
        return f"{cluster}: {CLUSTER_NAMES[cluster]}"
    return str(cluster)

def scatter_chart(df, scaled):
    """Create the PCA scatter chart."""
    pca = PCA(n_components=2)
    xy = pca.fit_transform(scaled)
    var = pca.explained_variance_ratio_ * 100  # % of the data's variation each axis shows

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.get_cmap("tab10")

    for cluster in sorted(df[KMEANS_COL].unique()):
        mask = df[KMEANS_COL] == cluster
        ax.scatter(xy[mask, 0], xy[mask, 1], color=colours(cluster), s=25, label=cname(cluster))

    if DBSCAN_COL in df.columns:
        noise = df[DBSCAN_COL] == -1
        ax.scatter(xy[noise, 0], xy[noise, 1], marker="x", c="black", s=60, label="DBSCAN noise (unusual)")

    ax.legend(loc="best", fontsize=8)  # one legend: clusters, plus the black X marks
    ax.set_xlabel(f"PCA component 1 ({var[0]:.0f}% of variation)")
    ax.set_ylabel(f"PCA component 2 ({var[1]:.0f}% of variation)")
    ax.set_title("LGA-year clusters (K-Means colours, PCA view)")

    plt.tight_layout()
    fig.savefig("figures/chart_cluster_scatter.png", dpi=200)
    plt.close(fig)

def profile_chart(df):
    """"Show the average feature values for each cluster."""
    means = df.groupby(KMEANS_COL)[FEATURES].mean()
    sizes = df[KMEANS_COL].value_counts().sort_index()

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3.2 * len(FEATURES), 4))

    for ax, feature in zip(axes, FEATURES):
        ax.bar([f"Cluster {cluster}\n(n={sizes[cluster]})" for cluster in means.index], means[feature], color=[plt.get_cmap("tab10")(cluster) for cluster in means.index])
        ax.set_title(feature, fontsize=9)

    plt.tight_layout()
    fig.savefig("figures/chart_cluster_profile.png", dpi=200)
    plt.close(fig)

    print("Cluster averages:\n", means.round(2).to_string())

def silhouette_chart(scaled):
    """Plot the silhouette scores for different numbers of clusters."""
    if os.path.exists(SCORES_FILE):
        scores = pd.read_csv(SCORES_FILE)
        ks = scores["k"].tolist() 
        sil = scores["silhouette_score"].tolist()
    else:
        ks = list(range(2, 9))
        sil = [
            silhouette_score(
                scaled, 
                KMeans(n_clusters=k, random_state=20).fit_predict(scaled))
               for k in ks]
        
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sil, marker="o")

    best = ks[sil.index(max(sil))]

    ax.annotate(f"highest: k={best} ({max(sil):.2f})", (best, max(sil)),
                textcoords="offset points", xytext=(85, 60), ha="center", arrowprops=dict(arrowstyle="->"))
    
    ax.set_ylim(0, 0.5)  # start at 0 so small differences are not made to look big
    ax.set_title("Silhouette score by number of clusters (k)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")

    plt.tight_layout()
    fig.savefig("figures/chart_cluster_silhouette.png", dpi=200)
    plt.close(fig)
    
    print("Silhouette by k:", {k: round(v, 2) for k, v in zip(ks, sil)})

def main():
    df = pd.read_csv(CSV_FILE)
    scaled = StandardScaler().fit_transform(df[FEATURES])

    print(f"{len(df)} rows, {df[LGA_COL].nunique()} LGAs")

    scatter_chart(df, scaled)
    profile_chart(df)
    silhouette_chart(scaled)

     # Check whether each LGA stays in the same cluster across all years
    stable = (df.groupby(LGA_COL)[KMEANS_COL].nunique() == 1).sum()
    print(f"{stable} of {df[LGA_COL].nunique()} LGAs stay in one cluster in all years")
    if DBSCAN_COL in df.columns:
        print("DBSCAN noise rows:", int((df[DBSCAN_COL] == -1).sum()))

if __name__ == "__main__":
    main()
