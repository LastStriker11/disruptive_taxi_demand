import numpy as np

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
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            r2_scores[i] = r2

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