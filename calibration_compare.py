#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
from sklearn.feature_selection import r_regression

from src.clustering import DTWClustering
from src.resilience_model import resilience_curve, general_logistic, loss_logistic
from src.analysis import compute_r2_smape_per_sample
from src.miscs import seed_everything

seed_everything(11)
sce = 'snowstorm'
if sce == 'covid':
    k = 4
else:
    k = 3
odtrips = np.load(f'results/{sce}_demand_normalized.npy')
odtrips[np.isnan(odtrips)] = 0
model = DTWClustering(odtrips, k)
model.clusters, model.centroids, cluster_params, all_losses = pickle.load(open(f'results/results_{sce}.pkl', 'rb'))

cluster_params_lf = []
for c in range(k):
    data = odtrips[model.clusters[c]]
    time = np.arange(data.shape[1])
    # Initial guess
    init_params = [0.5, 0.5, 1.0, 10.0]
    res = minimize(loss_logistic, init_params,
               args=(data, time), method='Nelder-Mead')

    best_params = res.x
    print("Fitted global parameters:", best_params)
    cluster_params_lf.append(best_params)
#%%
# Find the best-fitting time series
all_losses_lf = []
for c in range(k):
    best_params = cluster_params_lf[c]
    data = odtrips[model.clusters[c]]
    loss_cluster = []
    for i, series in enumerate(data):
        mu = np.mean(series)
        sigma = np.std(series)
        pred = general_logistic(time, mu, sigma, best_params)
        mse = np.mean((series - pred)**2)  # using mean instead of sum for comparability
        loss_cluster.append((i, mse))

    loss_cluster.sort(key=lambda x: x[1])
    all_losses_lf.append(loss_cluster)
#%%
# Calculate calibration accuracy for ILRF
all_r2 = []
all_mape = []
for c in range(k):
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
        mask = den == 0
        smape = np.empty_like(den)
        smape[mask] = np.nan
        smape[~mask] = np.abs(y_t[~mask] - y_p[~mask]) / den[~mask]

        smape = np.mean(smape) * 100
        mape_scores[i] = smape

    all_r2.append(r2_scores)
    all_mape.append(mape_scores)
# %%
# Calculate calibration accuracy for logistic functions
all_r2_lf = []
all_mape_lf = []
for c in range(k):
    cluster_curves = []
    data = odtrips[model.clusters[c]]
    r2_scores = np.zeros(len(data))
    mape_scores = np.zeros(len(data))
    for i, _ in enumerate(data):
        y_t = data[i]
        mu, sigma = np.mean(y_t), np.std(y_t)
        y_p = general_logistic(time, mu, sigma, cluster_params_lf[c])
        # ---- R^2 for sample i ----
        # ss_res = np.sum((y_t - y_p) ** 2)
        # ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        # r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        # r2_scores[i] = r2
        r2_scores[i] = np.sqrt((r_regression(y_p.reshape(-1, 1), y_t))**2)

        # ---- MAPE for sample i ----
        den = (np.abs(y_t) + np.abs(y_p)) / 2
        # Avoid division by zero (cases where both y_t and y_p are zero)
        mask = den < 0.1
        smape = np.empty_like(den)
        smape[mask] = np.nan
        smape[~mask] = np.abs(y_t[~mask] - y_p[~mask]) / den[~mask]

        smape = np.nanmean(smape) * 100
        mape_scores[i] = smape

    all_r2_lf.append(r2_scores)
    all_mape_lf.append(mape_scores)
#%%
n_od = 0
loss_mean = np.zeros(k)
for c in range(k):
        df_c = pd.DataFrame(all_losses[c])
        n_c = len(df_c)
        loss_mean_c = np.nanmean(df_c.iloc[:,-1])
        loss_mean[c] = loss_mean_c
        n_od += n_c

n_od = 0
loss_mean_lf = np.zeros(k)
for c in range(k):
    df_c = pd.DataFrame(all_losses_lf[c])
    n_c = len(df_c)
    loss_mean_c = np.nanmean(df_c.iloc[:,-1])
    loss_mean_lf[c] = loss_mean_c
    n_od += n_c
# %%
r2_sum = 0
r2_sum_lf = 0
mape_sum = 0
mape_sum_lf = 0
loss_sum = 0
loss_sum_lf = 0
for c in range(k):
    # ILRF
    r2c = all_r2[c]
    finite_mean = np.mean(r2c[np.isfinite(r2c)])
    r2c[np.isinf(r2c)] = finite_mean
    r2_sum += r2c.sum()
    mape_sum += all_mape[c].sum()
    loss_sum += loss_mean[c].sum()
    # Logistic
    r2c = all_r2_lf[c]
    finite_mean = np.mean(r2c[np.isfinite(r2c)])
    r2c[np.isinf(r2c)] = finite_mean
    r2_sum_lf += r2c.sum()
    mape_sum_lf += all_mape_lf[c].sum()
    loss_sum_lf += loss_mean_lf[c].sum()
    
print("==============ILRF metrics===============")
print(f"Avg $R^2$: ", r2_sum/odtrips.shape[0])
print(f"Avg SMAPE: ", mape_sum/odtrips.shape[0])
print(f"Avg loss: ", loss_sum/odtrips.shape[0])
print("==============logistic metrics===============")
print(f"Avg $R^2$: ", r2_sum_lf/odtrips.shape[0])
print(f"Avg SMAPE: ", mape_sum_lf/odtrips.shape[0])
print(f"Avg loss: ", loss_sum_lf/odtrips.shape[0])
#%%
time = range(len(model.centroids[0]))
ymax = 1.5
cmap = plt.cm.Reds
colors = [cmap((k+1-i)/(k+1)) for i in range(k+1)]
def plot_best_fits2(c, model, cluster_params, all_losses, cluster_params_lf, all_losses_lf):
    data = odtrips[model.clusters[c]]
    best_params = cluster_params[c]
    best_params_lf = cluster_params_lf[c]
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
        fitted_lf = general_logistic(time, mu, sigma, best_params_lf)
        
        ax[i].scatter(time, series, edgecolors='grey', facecolors='white', s=10)
        ax[i].plot(time, fitted, '-', label='ILRF', linewidth=2, color=colors[c])
        ax[i].plot(time, fitted_lf, ':', label='Logistic', linewidth=2, color=colors[c])
        ax[i].spines[['left','right','top','bottom']].set_visible(False)
        ax[i].set_xlim(0,40)
        ax[i].set_ylim(0,ymax)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].text(s=f"C{c+1}S{i+1}", x=15, y=1.2, color="#001BB7")
    ax[0].set_yticks([0,0.5,1,1.5])
    ax[0].spines['left'].set_visible(True)
    fig.savefig(f"figures/fit_cluster{c+1}_lf_ilrf.pdf", bbox_inches='tight')

for c in range(4):
    plot_best_fits2(c, model, cluster_params, all_losses, cluster_params_lf, all_losses_lf)
# %%
