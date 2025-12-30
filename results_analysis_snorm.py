#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import pickle
from src.clustering import DTWClustering
from src.resilience_model import resilience_curve, loss
#%%
# load data and model
odtrips = np.load('results/snowstorm_demand_normalized_v2.npy')
odtrips = odtrips[~np.isnan(odtrips).any(axis=1)]
model = DTWClustering(odtrips, 3)
model.clusters, model.centroids, cluster_params, all_losses = pickle.load(open('results/results_snowstorm_v2.pkl', 'rb'))
# reorder clusters: 0,1,2 -> 1,0,2
model.clusters = {0: model.clusters[1],
                  1: model.clusters[0],
                  2: model.clusters[2]}
model.centroids = {0: model.centroids[1],
                   1: model.centroids[0],
                   2: model.centroids[2]}
cluster_params = {0: cluster_params[1],
                  1: cluster_params[0],
                  2: cluster_params[2]}
all_losses = {0: all_losses[1],
               1: all_losses[0],
               2: all_losses[2]}
#%%
xmax = len(model.centroids[0])
ymax = 1.5
time = range(xmax)
cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]
# %%
def plot_best_fits(c, model, cluster_params, all_losses):
    data = odtrips[model.clusters[c]]
    best_params = cluster_params[c]
    loss_cluster = all_losses[c]
    series_indices = [s for s, _ in loss_cluster[:10]]  # first 10 series with best fit
    time = np.arange(data.shape[1])

    ncols = 5
    nrows = 2
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12,4))
    for i, s in enumerate(series_indices):
        series = data[s]
        mu = np.mean(series)
        sigma = np.std(series)
        fitted = resilience_curve(time, mu, sigma, best_params)
        
        ax[int(i/ncols),i%ncols].tick_params(direction='in', top=True, right=True, which='both', width=1)
        ax[int(i/ncols),i%ncols].spines[['bottom','top','left','right']].set_linewidth(1)
        ax[int(i/ncols),i%ncols].yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[int(i/ncols),i%ncols].xaxis.set_minor_locator(tck.AutoMinorLocator())
        ax[int(i/ncols),i%ncols].scatter(time, series, edgecolors='grey', facecolors='white', s=10)
        ax[int(i/ncols),i%ncols].plot(time, fitted, '-', label='Fitted', linewidth=2)
        ax[int(i/ncols),i%ncols].grid(True, linestyle='--', color='gray', alpha=0.5)
        if int(i/ncols) == 1:
            ax[int(i/ncols),i%ncols].set_xlabel("Time (weeks)")
        if i%ncols == 0:
            ax[int(i/ncols),i%ncols].set_ylabel("Normalized # of trips")
        else:
            ax[int(i/ncols),i%ncols].set_yticklabels([])
        ax[int(i/ncols),i%ncols].grid(True)
        ax[int(i/ncols),i%ncols].set_xlim(0,xmax)
        ax[int(i/ncols),i%ncols].set_ylim(0,ymax)
    fig.savefig(f"figures/fit_snow_cluster{c+1}.pdf", bbox_inches='tight')

for c in range(3):
    plot_best_fits(c, model, cluster_params, all_losses)
#%%
def plot_best_fits2(c, model, cluster_params, all_losses):
    data = odtrips[model.clusters[c]]
    best_params = cluster_params[c]
    loss_cluster = all_losses[c]
    series_indices = [s for s, _ in loss_cluster[:10]]  # first 10 series with best fit
    time = np.arange(data.shape[1])

    ncols = 10
    nrows = 1
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20,2))
    for i, s in enumerate(series_indices):
        series = data[s]
        mu = np.mean(series)
        sigma = np.std(series)
        fitted = resilience_curve(time, mu, sigma, best_params)
        
        ax[i].scatter(time, series, edgecolors='grey', facecolors='white', s=10)
        ax[i].plot(time, fitted, '-', label='Fitted', linewidth=2, color=colors[c])
        ax[i].spines[['left','right','top','bottom']].set_visible(False)
        ax[i].set_xlim(0,xmax)
        ax[i].set_ylim(0,ymax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].text(s=f"C{c+1}S{i+1}", x=4, y=1.3, color="#001BB7")
    ax[0].set_yticks([0,0.5,1,1.5])
    ax[0].spines['left'].set_visible(True)
    fig.savefig(f"figures/fit_snow_cluster{c+1}_curves.pdf", bbox_inches='tight')

for c in range(3):
    plot_best_fits2(c, model, cluster_params, all_losses) 
# %%
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(7,1.8))

cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]
for c in range(3):
    ax[c].tick_params(direction='in', top=True, right=True, which='both', width=1)
    ax[c].spines[['bottom','top','left','right']].set_linewidth(1)
    ax[c].yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[c].xaxis.set_minor_locator(tck.AutoMinorLocator())

    cluster_curves = []
    data = odtrips[model.clusters[c]]
    # Compute fitted curves for all series in the cluster
    for i, _ in enumerate(data):
        mu, sigma = np.mean(data[i]), np.std(data[i])
        curve = resilience_curve(time, mu, sigma, cluster_params[c])
        cluster_curves.append(curve)
        # ax[c].scatter(np.arange(len(data[i])), data[i], color='grey', alpha=0.3, s=20)

    cluster_curves = np.array(cluster_curves)
    mean_curve = np.mean(cluster_curves, axis=0)
    lower_bound = np.min(cluster_curves, axis=0)
    upper_bound = np.max(cluster_curves, axis=0)

    # Plot the DTW center
    ax[c].plot(np.arange(len(model.centroids[c])), model.centroids[c], color='tab:blue', linestyle='--', linewidth=1)
    # Plot mean and bounds
    ax[c].plot(time, mean_curve, color=colors[c], linewidth=1.5, label=f'Cluster {c+1}')
    ax[c].fill_between(time, lower_bound, upper_bound, color=colors[c], alpha=0.4)
    # ax[c].text(15, ymax-0.2, f'Cluster {c+1}', color='tab:blue')

    ax[c].set_xlabel("Time (weeks)")
    ax[c].set_xlim(0,xmax-1)
    ax[c].set_ylim(0,ymax)
    if c != 0:
        ax[c].set_yticklabels([])
    ax[c].grid(True, linestyle='--', alpha=0.4)
ax[0].set_ylabel("Normalized # of trips")
fig.savefig(f"figures/resilience_patterns_snow.pdf", bbox_inches="tight")
# %%
