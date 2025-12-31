#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from sklearn.feature_selection import r_regression

import pickle
from src.clustering import DTWClustering
from src.resilience_model import resilience_curve, loss
from src.analysis import compute_r2_smape_per_sample
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
# %%
time = range(len(model.centroids[0]))
ymax = 1.5
cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]
# %%
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,2.2))
for centroid_key in model.clusters:
    ax[centroid_key].tick_params(direction='in', top=True, right=True, which='both', width=1)
    ax[centroid_key].spines[['bottom','top','left','right']].set_linewidth(1)
    ax[centroid_key].yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax[centroid_key].xaxis.set_minor_locator(tck.AutoMinorLocator())

    ax[centroid_key].vlines(x=[6,12], ymin=0, ymax=ymax, colors='#124170', linestyles='-', linewidth=1.5)
    ax[centroid_key].fill_betweenx(y=[0,ymax], x1=6, x2=12, color='pink', alpha=0.5)
    ax[centroid_key].fill_betweenx(y=[0,ymax], x1=12, x2=40, color='lightgreen', alpha=0.5)
    for series_index in model.clusters[centroid_key]:
        series = odtrips[series_index]
        ax[centroid_key].scatter(np.arange(len(series)), series, color='grey', alpha=0.3)
    ax[centroid_key].plot(np.arange(len(series)), model.centroids[centroid_key], color='tab:blue')
    ax[centroid_key].text(15, ymax-0.2, f'Cluster {centroid_key+1}', color='tab:blue')

    # if centroid_key%2 == 2:
    ax[centroid_key].set_xlabel("Time (weeks)")
    # ax[centroid_key].set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax[centroid_key].set_xlim(0,40)
    ax[centroid_key].set_ylim(0,ymax)
    if centroid_key != 0:
        ax[centroid_key].set_yticklabels([])
ax[0].set_ylabel("Normalized # of trips")
plt.tight_layout()
fig.savefig(f"figures/clustering.pdf", bbox_inches='tight')
#%%
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
        ax[int(i/ncols),i%ncols].set_xlim(0,40)
        ax[int(i/ncols),i%ncols].set_ylim(0,ymax)
    fig.savefig(f"figures/resilience_fit_cluster{c+1}.pdf", bbox_inches='tight')

for c in range(4):
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
        ax[i].set_xlim(0,40)
        ax[i].set_ylim(0,ymax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].text(s=f"C{c+1}S{i+1}", x=15, y=1.2, color="#001BB7")
    ax[0].set_yticks([0,0.5,1,1.5])
    ax[0].spines['left'].set_visible(True)
    fig.savefig(f"figures/resilience_fit_cluster{c+1}_curves.pdf", bbox_inches='tight')

for c in range(4):
    plot_best_fits2(c, model, cluster_params, all_losses)
# %%
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,1.8))

clusters = [0,1,2,3]
cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]
for c in clusters:
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
    ax[c].text(15, ymax-0.2, f'Cluster {c+1}', color='tab:blue')

    ax[c].set_xlabel("Time (weeks)")
    # ax[c].set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax[c].set_xlim(0,40)
    ax[c].set_ylim(0,ymax)
    if c != 0:
        ax[c].set_yticklabels([])
    ax[c].grid(True, linestyle='--', alpha=0.4)
ax[0].set_ylabel("Normalized # of trips")
fig.savefig(f"figures/resilience_patterns.pdf", bbox_inches="tight")
# %%
# compute r2 and smape
r2_per_sample, mape_per_sample = compute_r2_smape_per_sample(odtrips, model.centroids, model.clusters)

# These should each be length n_clusters
n_od = 0
r2_mean = np.zeros(4)
r2_std = np.zeros(4)
mape_mean = np.zeros(4)
mape_std = np.zeros(4)
loss_mean = np.zeros(4)
loss_std = np.zeros(4)
for c in range(4):
    df_c = pd.DataFrame(all_losses[c])
    n_c = len(df_c)
    r2c = r2_per_sample[df_c.iloc[:,0].tolist()]
    mapec = mape_per_sample[df_c.iloc[:,0].tolist()]
    
    r2_mean_c = r2c.mean()
    r2_std_c = r2c.std()
    mape_mean_c = np.nanmean(mapec)
    mape_std_c = np.nanstd(mapec)
    
    loss_mean_c = np.nanmean(df_c.iloc[:,-1])
    loss_std_c = np.nanstd(df_c.iloc[:,-1])

    n_od += n_c
    r2_mean[c] = r2_mean_c
    r2_std[c] = r2_std_c
    mape_mean[c] = mape_mean_c
    mape_std[c] = mape_std_c
    loss_mean[c] = loss_mean_c
    loss_std[c] = loss_std_c
#%%
n_clusters = 4
x = np.arange(n_clusters)
width = 0.35  # bar width
# -------------------------------------
fig, ax1 = plt.subplots(figsize=(4.5,2.5))
ax1.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax1.spines[['bottom','top','left']].set_linewidth(1.5)
ax1.spines["right"].set_visible(False)

# --- Left axis (Abs. Pearson) ---
ax1.set_xlabel("Cluster")
ax1.set_ylabel(r"Absolute Pearson correlation", color="#001BB7")
bars_r2 = ax1.bar(x - width/2, r2_mean, width, yerr=r2_std, 
                  capsize=5, label=r"Abs. Pearson",
                  edgecolor="#001BB7", facecolor="#AAC4F5", ecolor="#001BB7")
ax1.tick_params(axis="y", colors="#001BB7")
ax1.spines["left"].set_color("#001BB7")
ax1.set_ylim([0, 1.25])
ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])

# --- Right axis (MAPE) ---
ax2 = ax1.twinx()
ax2.set_ylabel("SMAPE (%)", color="#A72703")
bars_mape = ax2.bar(x + width/2, mape_mean, width, yerr=mape_std, capsize=5, label="SMAPE",
                    edgecolor="#A72703", facecolor="#FFF2EF", ecolor="#A72703")
ax2.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax2.spines[['bottom','right']].set_linewidth(1.5)
ax2.spines[['top','left']].set_visible(False)
ax2.tick_params(axis="y", colors="#A72703")
ax2.spines["right"].set_color("#A72703")
ax2.set_ylim([0,50])

# X axis labels
ax1.set_xticks(x)
ax1.set_xticklabels([f"Cluster {i+1}" for i in range(n_clusters)])
handles = [bars_r2, bars_mape]
labels = [r"Abs. Pearson", "SMAPE"]
ax1.legend(handles, labels, ncols=2, loc="upper right")

fig.savefig("figures/pearson_mape_covid.pdf", bbox_inches="tight")
#%%
fig, ax = plt.subplots(figsize=(3,2.5))
ax.tick_params(direction='in', top=True, right=True, which='both', width=1)
ax.spines[['bottom','top','left','right']].set_linewidth(1)
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

cmap = plt.cm.Blues
colors = [cmap((4-i)/4) for i in range(4)]
ls = ['-', '--', '-.', ':']
for c in range(4):
    df_c = pd.DataFrame(all_losses[c]) 
    r2c = r2_per_sample[df_c.iloc[:,0].tolist()]
    kde = gaussian_kde(r2c)
    x_vals = np.linspace(r2c.min(), r2c.max(), 200)
    density = kde(x_vals)
    ax.plot(x_vals, density, label=f"Cluster {c+1}", linestyle=ls[c])

leg = ax.legend(loc="upper left", frameon=True)
ax.set_xlabel(r"Absolute Pearson correlation")
ax.set_ylabel("Probability density")
ax.set_xlim([0.77,1])
ax.set_ylim([0,13])

fig.savefig("figures/dist_pearson_covid.pdf", bbox_inches="tight")
#%%
fig, ax = plt.subplots(figsize=(3,2.5))
ax.tick_params(direction='in', top=True, right=True, which='both', width=1)
ax.spines[['bottom','top','left', 'right']].set_linewidth(1)
ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

cmap = plt.cm.Reds
colors = [cmap((4-i)/4) for i in range(4)]

for c in range(4):
    df_c = pd.DataFrame(all_losses[c])
    mapec = mape_per_sample[df_c.iloc[:,0].tolist()]
    mapec = mapec[~np.isnan(mapec)]
    # mapec = mapec[mapec<100]
    kde = gaussian_kde(mapec)
    x_vals = np.linspace(0, 100, 200)
    density = kde(x_vals)

    ax.plot(x_vals, density, label=f"Cluster {c+1}", linestyle=ls[c])
ax.legend()
ax.set_xlabel("SMAPE (%)")
ax.set_ylabel("Probability density")
ax.set_xlim([0,100])
ax.set_ylim([0,0.05])
fig.savefig("figures/dist_mape_covid.pdf", bbox_inches="tight")
# %%
# Calculate calibration accuracy
all_r2 = []
all_mape = []
rec_25 = []
rec_40 = []
for c in range(4):
    cluster_curves = []
    data = odtrips[model.clusters[c]]
    r2_scores = np.zeros(len(data))
    mape_scores = np.zeros(len(data))
    for i, _ in enumerate(data):
        y_t = data[i]
        mu, sigma = np.mean(y_t), np.std(y_t)
        y_p = resilience_curve(time, mu, sigma, cluster_params[c])
        # ---- R^2 for sample i ----
        # ss_res = np.sum((y_t - y_p) ** 2)
        # ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        # r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        # r2_scores[i] = r2
        r2_scores[i] = np.sqrt((r_regression(y_p.reshape(-1, 1), y_t))**2)

        # ---- MAPE for sample i ----
        den = (np.abs(y_t) + np.abs(y_p)) / 2
        # Avoid division by zero (cases where both y_t and y_p are zero)
        # mask = den == 0
        mask = den < 0.1
        smape = np.empty_like(den)
        smape[mask] = np.nan
        smape[~mask] = np.abs(y_t[~mask] - y_p[~mask]) / den[~mask]

        smape = np.mean(smape) * 100
        mape_scores[i] = smape

    all_r2.append(r2_scores)
    all_mape.append(mape_scores)
    rec_25.append(data[:,24].mean())
    rec_40.append(data[:,-1].mean())
# %%
for c in range(4):
    print(f"Absolute Pearson of Cluster {c+1}: ", all_r2[c].mean())
    # print(f"Std $R^2$ of Cluster {c+1}: ", all_r2[c].std())
    print(f"Avg SMAPE of Cluster {c+1}: ", np.nanmean(all_mape[c]))
    # print(f"Std SMAPE of Cluster {c+1}: ", np.nanstd(all_mape[c]))
    print(f"RMSE of Cluster {c+1}: ", np.sqrt(loss_mean[c]))
    # print(f"Std loss of Cluster {c+1}: ", loss_std[c])
    print(f"Value at week 25: ", rec_25[c])
    print(f"Value at week 40: ", rec_40[c])
    print("Parameters: ", cluster_params[c])
    print("---------------------------------------------")
# %%
# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(4,2.2))
for centroid_key in model.clusters:
    ax[int(centroid_key/2),centroid_key%2].spines[['left','right','top','bottom']].set_visible(False)

    for series_index in model.clusters[centroid_key][:5]:
        series = odtrips[series_index]
        ax[int(centroid_key/2),centroid_key%2].plot(np.arange(len(series)), series, color='grey', alpha=0.3)
    ax[int(centroid_key/2),centroid_key%2].plot(np.arange(len(series)), model.centroids[centroid_key], color='tab:blue')

    # if centroid_key%2 == 2:
    ax[int(centroid_key/2),centroid_key%2].set_xticks([])
    ax[int(centroid_key/2),centroid_key%2].set_yticks([])
    # ax[centroid_key].set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax[int(centroid_key/2),centroid_key%2].set_xlim(0,40)
    ax[int(centroid_key/2),centroid_key%2].set_ylim(0,ymax)
plt.tight_layout()
fig.savefig(f"figures/clustering_sample.pdf", bbox_inches='tight')
#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1.8,1.8))

cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]
ax.tick_params(direction='in', top=True, right=True, which='both', width=1)
ax.spines[['bottom','top','left','right']].set_visible(False)

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
ax.plot(np.arange(len(model.centroids[c])), model.centroids[c], color='tab:blue', linestyle='--', linewidth=1)
# Plot mean and bounds
ax.plot(time, mean_curve, color=colors[c], linewidth=1.5, label=f'Cluster {c+1}')
# ax.fill_between(time, lower_bound, upper_bound, color=colors[c], alpha=0.4)

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0,40)
ax.set_ylim(0,ymax)
fig.savefig(f"figures/pattern4.pdf", bbox_inches="tight")
#%%
cluster_sample = [cluster_curves[0], lower_bound, upper_bound]
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(1.8,6))

cmap = plt.cm.Reds
colors = [cmap((5-i)/5) for i in range(5)]

for s in range(3):
    ax[s].tick_params(direction='in', top=True, right=True, which='both', width=1)
    ax[s].spines[['bottom','top','left','right']].set_visible(False)
    # Plot mean and bounds
    ax[s].plot(time, cluster_sample[s], color=colors[c], linewidth=1.5, label=f'Cluster {c+1}')
    # ax.fill_between(time, lower_bound, upper_bound, color=colors[c], alpha=0.4)

    ax[s].set_xticks([])
    ax[s].set_yticks([])
    ax[s].set_xlim(0,40)
    ax[s].set_ylim(0,ymax+0.2)
fig.savefig(f"figures/sample4.pdf", bbox_inches="tight")