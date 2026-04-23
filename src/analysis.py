import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression
from dtaidistance import dtw
from tqdm import tqdm
# Inputs:
# y_true: (n_samples, n_time_steps)
# centroids: (n_clusters, n_time_steps)
# cluster_dict: e.g., {"cluster 0": [0,2,3], "cluster 1": [1,4]}

def compute_r2_smape_per_sample(y_true, centroids, cluster_dict):
    n_samples = y_true.shape[0]
    
    # Arrays to store sample-wise R^2 and MAPE
    r2_scores = np.zeros(n_samples)
    mape_scores = np.zeros(n_samples)

    # Iterate cluster by cluster
    for c_name, sample_indices in cluster_dict.items():
        # Extract integer cluster index if needed (e.g., "cluster 0")
        cluster_idx = int(c_name)
        
        # Corresponding centroid time-series
        y_pred_cluster = centroids[cluster_idx]  # shape: (n_time_steps,)
        
        for i in sample_indices:
            y_t = y_true[i]              # shape: (n_time_steps,)
            y_p = np.array(y_pred_cluster)         # shape: (n_time_steps,)

            # ---- R^2 for sample i ----
            # ss_res = np.sum((y_t - y_p) ** 2)
            # ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            # r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            # r2_scores[i] = r2
            # r2_scores[i] = r2_score(y_t, y_p)
            r2_scores[i] = np.sqrt((r_regression(y_p.reshape(-1, 1), y_t))**2)

            # ---- MAPE for sample i ----
            den = (np.abs(y_t) + np.abs(y_p)) / 2

            # Avoid division by zero (cases where both y_t and y_p are zero)
            mask = den == 0
            smape = np.empty_like(den)
            smape[mask] = np.nan
            smape[~mask] = np.abs(y_t[~mask] - y_p[~mask]) / den[~mask]

            smape = np.mean(smape) * 100
            mape_scores[i] = smape

    return r2_scores, mape_scores


def silhouette_score_dtw_memory_efficient(X, labels):
    """
    Memory-efficient silhouette score calculation using DTW.
    """
    n_samples = len(X)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    
    silhouette_values = []
    
    # Calculate silhouette score for each sample
    for i in tqdm(range(n_samples), desc="Computing silhouette scores"):
        a_i = 0  # Mean distance to same cluster
        same_cluster_count = 0
        
        b_i = float('inf')  # Min mean distance to other clusters
        
        # Calculate distances to all other points
        distances_to_others = []
        
        for j in range(n_samples):
            if i != j:
                dist = dtw.distance(X[i], X[j])
                distances_to_others.append((dist, labels[j]))
        
        # Calculate a(i): mean distance to points in same cluster
        same_cluster_distances = [dist for dist, lbl in distances_to_others if lbl == labels[i]]
        if same_cluster_distances:
            a_i = np.mean(same_cluster_distances)
        
        # Calculate b(i): min mean distance to other clusters
        for other_label in unique_labels:
            if other_label != labels[i]:
                other_cluster_distances = [dist for dist, lbl in distances_to_others if lbl == other_label]
                if other_cluster_distances:
                    mean_distance = np.mean(other_cluster_distances)
                    b_i = min(b_i, mean_distance)
        
        # Calculate silhouette for sample i
        if b_i == float('inf'):
            s_i = 0
        else:
            s_i = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_values.append(s_i)
    
    return np.mean(silhouette_values)