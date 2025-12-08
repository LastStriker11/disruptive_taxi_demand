#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.clustering import benchmark_dtw_vs_euclidean
from src.analysis import compute_r2_smape_per_sample
from src.miscs import seed_everything

seed_everything(11)

sce = 'covid'
odtrips = np.load(f'results/{sce}_demand_normalized.npy')
odtrips[np.isnan(odtrips)] = 0
max_iters = 100
if sce == 'covid':
    k = 4
else:
    k = 3
# perform clustering

all_metrics_dtw = []
all_metrics_euc = []
for i in range(10):
    results = benchmark_dtw_vs_euclidean(
        data=np.array(odtrips),
        k=k,
        max_iters=max_iters,
        window_size=3,
        verbose=False
    )
    metrics_dtw = results["DTW"]["metrics"]
    metrics_euc = results["Euclidean"]["metrics"]

    centroid_dtw = results["DTW"]["centroids"]
    cluster_dtw = results["DTW"]["labels"]
    r2_per_sample, mape_per_sample = compute_r2_smape_per_sample(odtrips, centroid_dtw, cluster_dtw)
    metrics_dtw["r2"] = np.nanmean(r2_per_sample)
    metrics_dtw["smape"] = np.nanmean(mape_per_sample)

    centroid_euc = results["Euclidean"]["centroids"]
    cluster_euc = results["Euclidean"]["labels"]
    r2_per_sample, mape_per_sample = compute_r2_smape_per_sample(odtrips, centroid_euc, cluster_euc)
    metrics_euc["r2"] = np.nanmean(r2_per_sample)
    metrics_euc["smape"] = np.nanmean(mape_per_sample)

    all_metrics_dtw.append(metrics_dtw)
    all_metrics_euc.append(metrics_euc)
df_dtw = pd.DataFrame(all_metrics_dtw)
df_euc = pd.DataFrame(all_metrics_euc)
#%%
# save results
# df_dtw = pd.read_csv(f"results/all_metrics_dtw_covid.csv")
# df_euc = pd.read_csv(f"results/all_metrics_euc_covid.csv")
df_dtw.to_csv(f"results/all_metrics_dtw_{sce}.csv", index=False)
df_euc.to_csv(f"results/all_metrics_euc_{sce}.csv", index=False)
#%%
mean_dtw = df_dtw.mean().values
mean_dtw = mean_dtw[[0,3,4]]
mean_euc = df_euc.mean().values
mean_euc = mean_euc[[0,3,4]]
std_dtw = df_dtw.std().values
std_dtw = std_dtw[[0,3,4]]
std_euc = df_euc.std().values
std_euc = std_euc[[0,3,4]]
print(mean_dtw)
print(mean_euc)
print(std_dtw)
print(std_euc)
# %%