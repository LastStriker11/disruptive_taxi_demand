#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression

import pickle
from src.clustering import DTWClustering
from src.resilience_model import resilience_curve_sa
#%%
# load data and model
odtrips = np.load('results/covid_demand_normalized.npy')
model = DTWClustering(odtrips, 4)
model.clusters, model.centroids, cluster_params, all_losses = pickle.load(open('results/results_covid.pkl', 'rb'))
# reorder clusters: 0,1,2,3 -> 2,0,1,3
model.clusters = {0: model.clusters[2],
                  1: model.clusters[0],
                  2: model.clusters[1],
                  3: model.clusters[3]}
model.centroids = {0: model.centroids[2],
                   1: model.centroids[0],
                   2: model.centroids[1],
                   3: model.centroids[3]}
cluster_params = {0: cluster_params[2],
                  1: cluster_params[0],
                  2: cluster_params[1],
                  3: cluster_params[3]}
all_losses = {0: all_losses[2],
               1: all_losses[0],
               2: all_losses[1],
               3: all_losses[3]}

#%%
apcc_all_avg = []
apcc_all_std = []
factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
for factor in factors:
    apcc_factor = []
    for c in range(4):
        data = odtrips[model.clusters[c]]
        best_params = cluster_params[c]
        loss_cluster = all_losses[c]
        series_indices = [s for s, _ in loss_cluster[:]]  # first 10 series with best fit
        time = np.arange(data.shape[1])
        apcc_cluster = []
        for i, s in enumerate(series_indices):
            series = data[s]
            mu = np.mean(series)
            sigma = np.std(series)
            mu_Ad = mu * factor
            sigma_Ad = sigma * factor
            mu_Ar = mu
            sigma_Ar = sigma
            fitted = resilience_curve_sa(time, mu_Ad, sigma_Ad, mu_Ar, sigma_Ar, best_params)
            apcc_od = np.sqrt((r_regression(fitted.reshape(-1, 1), series))**2)
            apcc_cluster.append(apcc_od[0])
        apcc_factor.append(apcc_cluster)
    apcc_avg = [np.mean(apcc) for apcc in apcc_factor]
    apcc_std = [np.std(apcc) for apcc in apcc_factor]
    apcc_all_avg.append(apcc_avg)
    apcc_all_std.append(apcc_std)
# %%
cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines[['bottom','top','left', 'right']].set_linewidth(1.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

for c in range(4):
    apcc_avg = [apcc[c] for apcc in apcc_all_avg]
    apcc_std = [apcc[c] for apcc in apcc_all_std]
    ax.errorbar(factors, apcc_avg, yerr=apcc_std, label=f'Cluster {c+1}', 
                marker='o', capsize=3, linestyle='-', color=colors[c])
ax.set_xticks(factors)
ax.set_xticklabels(factors)
ax.set_xlabel(r'$A_d$ factor') 
ax.set_ylabel('Absolute Pearson Correlation Coefficient')
ax.legend()
fig.savefig("figures/apcc_sensitivity_Ad.pdf", bbox_inches='tight')
# %%
apcc_all_avg = []
apcc_all_std = []
factors = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2]
for factor in factors:
    apcc_factor = []
    for c in range(4):
        data = odtrips[model.clusters[c]]
        best_params = cluster_params[c]
        loss_cluster = all_losses[c]
        series_indices = [s for s, _ in loss_cluster[:]]  # first 10 series with best fit
        time = np.arange(data.shape[1])
        apcc_cluster = []
        for i, s in enumerate(series_indices):
            series = data[s]
            mu = np.mean(series)
            sigma = np.std(series)
            mu_Ad = mu
            sigma_Ad = sigma
            mu_Ar = mu * factor
            sigma_Ar = sigma * factor
            fitted = resilience_curve_sa(time, mu_Ad, sigma_Ad, mu_Ar, sigma_Ar, best_params)
            apcc_od = np.sqrt((r_regression(fitted.reshape(-1, 1), series))**2)
            apcc_cluster.append(apcc_od[0])
        apcc_factor.append(apcc_cluster)
    apcc_avg = [np.mean(apcc) for apcc in apcc_factor]
    apcc_std = [np.std(apcc) for apcc in apcc_factor]
    apcc_all_avg.append(apcc_avg)
    apcc_all_std.append(apcc_std)
# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax.spines[['bottom','top','left', 'right']].set_linewidth(1.5)
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

for c in range(4):
    apcc_avg = [apcc[c] for apcc in apcc_all_avg]
    apcc_std = [apcc[c] for apcc in apcc_all_std]
    ax.errorbar(factors, apcc_avg, yerr=apcc_std, label=f'Cluster {c+1}', 
                marker='o', capsize=3, linestyle='-', color=colors[c])
ax.set_xticks(factors)
ax.set_xticklabels(factors)
ax.set_xlabel(r'$A_r$ factor') 
ax.set_ylabel('Absolute Pearson Correlation Coefficient')
# ax.legend()
fig.savefig("figures/apcc_sensitivity_Ar.pdf", bbox_inches='tight')
# %%
