#%%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from src.clustering import DTWClustering
from src.resilience_model import resilience_curve, loss
#%%
# load data
odtrips = np.load('results/snowstorm_demand_normalized_v2.npy')
odtrips = odtrips[~np.isnan(odtrips).any(axis=1)]
# perform clustering
np.random.seed(11)
model = DTWClustering(odtrips, k=3, max_iters=100)
model.train()
#%%
time = range(len(model.centroids[0]))
ymax = 1.5
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7,2.2))
for centroid_key in model.clusters:
    ax[centroid_key].tick_params(direction='in', top=True, right=True, which='both', width=1)
    ax[centroid_key].spines[['bottom','top','left','right']].set_linewidth(1)
    ax[centroid_key].yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[centroid_key].xaxis.set_minor_locator(tck.AutoMinorLocator())

    # ax[centroid_key].vlines(x=[6,12], ymin=0, ymax=ymax, colors='#124170', linestyles='-', linewidth=1.5)
    # ax[centroid_key].fill_betweenx(y=[0,ymax], x1=6, x2=12, color='pink', alpha=0.5)
    # ax[centroid_key].fill_betweenx(y=[0,ymax], x1=12, x2=40, color='lightgreen', alpha=0.5)
    for series_index in model.clusters[centroid_key]:
        series = odtrips[series_index]
        ax[centroid_key].scatter(np.arange(len(series)), series, color='grey', alpha=0.3)
    ax[centroid_key].plot(np.arange(len(series)), model.centroids[centroid_key], color='tab:blue')
    # ax[centroid_key].text(15, ymax-0.2, f'Cluster {centroid_key+1}', color='tab:blue')

    # if centroid_key%2 == 2:
    ax[centroid_key].set_xlabel("Time (weeks)")
    # ax[centroid_key].set_yticks([0,0.2,0.4,0.6,0.8,1])
    # ax[centroid_key].set_xlim(0,40)
    ax[centroid_key].set_ylim(0,ymax)
    if centroid_key != 0:
        ax[centroid_key].set_yticklabels([])
ax[0].set_ylabel("Normalized # of trips")
plt.tight_layout()
# %%
cluster_params = []
for c in range(3):
    data = odtrips[model.clusters[c]]
    time = np.arange(data.shape[1])
    # Initial guess
    # [alpha_d, beta_d, k_d, v_d, 
    # alpha_r, beta_r, k_r, v_r, m, 
    # t_d, t_r, t_s, k_s]
    # params0 = np.array([-0.5, -0.5, 1.5, 2,
    #                     -0.5, -0.5, -0.1, 3.5, 0.02,
    #                     7.5, 20, 11, 0.5])
    # bounds = [(None,None), (None,None), (None,None), (None,None), 
    #           (None,None), (None,None), (None,None), (None,None), (None,None),
    #           (5,8), (12,40), (8,15), (0,None)]
    # if c == 2:
    params0 = np.array([-0.3, -0.3, 0, 2,
                        -0.3, -0.3, 0, 2, 0,
                        8, 12, 10, 0])
    if c == 1:
        params0 = np.array([-0.3, -0.3, 1.0, 2,
                        -0.3, -0.3, 0.5, 3.0, 0.02,
                        10, 13, 11, 0.5])
    if c == 2:
        params0 = np.array([-0.5, -0.5, 0.5, 2,
                        -0.3, -0.3, -0.1, 3, 0,
                        10, 10, 10, 1])
    bounds = [(None,None), (None,None), (None,None), (None,None), 
            (None,None), (None,None), (None,None), (None,None), (None,None),
            (None,None), (None,None), (None,None), (0,None)]
    res = minimize(loss, params0, args=(data, time), method='L-BFGS-B', bounds=bounds)

    best_params = res.x
    print("Fitted global parameters:", best_params)
    cluster_params.append(best_params)
#%%
# Find the best-fitting time series
all_losses = []
for c in range(3):
    best_params = cluster_params[c]
    data = odtrips[model.clusters[c]]
    loss_cluster = []
    for i, series in enumerate(data):
        mu = np.mean(series)
        sigma = np.std(series)
        pred = resilience_curve(time, mu, sigma, best_params)
        mse = np.mean((series - pred)**2)  # using mean instead of sum for comparability
        loss_cluster.append((i, mse))

    # Sort by MSE (ascending)
    loss_cluster.sort(key=lambda x: x[1])

    # Print the 10 best fits
    print("Top 10 best-fitting series (index, MSE):")
    for idx, mse in loss_cluster[:10]:
        print(f"Series {idx}: MSE = {mse:.6f}")
    all_losses.append(loss_cluster)
# %%
f = open('results/results_snowstorm_v2.pkl', 'wb')
pickle.dump([model.clusters, model.centroids, cluster_params, all_losses], f)
f.close()
# %%
