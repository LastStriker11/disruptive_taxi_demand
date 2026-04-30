# %%
# Euclidean + Logistic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.feature_selection import r_regression
import pickle

from src.clustering import DTWClustering
from src.clustering import benchmark_dtw_vs_euclidean
from src.miscs import seed_everything
from src.resilience_model import resilience_curve, general_logistic, loss_logistic, loss

seed_everything(11)
#%%
sce = 'snowstorm'
odtrips = np.load(f'results/{sce}_demand_normalized.npy')
odtrips[np.isnan(odtrips)] = 0
max_iters = 100
if sce == 'covid':
    k = 4
else:
    k = 3
#%%
all_metrics_dtw = []
all_metrics_euc = []
results = benchmark_dtw_vs_euclidean(
    data=np.array(odtrips),
    k=k,
    max_iters=max_iters,
    window_size=3,
    verbose=False
)

centroid_euc = results["Euclidean"]["centroids"]
cluster_euc = results["Euclidean"]["labels"]
#%%
cluster_params = []
for c in range(k):
    data = odtrips[cluster_euc[str(c)]]
    time = np.arange(data.shape[1])
    # Initial guess
    # [alpha_d, beta_d, k_d, v_d, alpha_r, beta_r, k_r, v_r, m, t_d, t_r, t_s, k_s]
    params0 = np.array([-0.5, -0.5, 1.5, 2,
                        -0.5, -0.5, -0.1, 3.5, 0.02,
                        7.5, 20, 11, 0.5])
    bounds = [(None,None), (None,None), (None,None), (None,None), 
              (None,None), (None,None), (None,None), (None,None), (None,None),
              (5,8), (12,40), (8,15), (0,None)]
    if c == 0:
        params0 = np.array([-0.3, -0.3, 1.0, 2,
                            -0.3, -0.3, -0.1, 3.0, 0.01,
                            7.5, 25, 12, 0.5])
    if c == 1:
        params0 = np.array([-0.3, -0.3, 1.0, 2,
                            -0.3, -0.3, -0.1, 3.0, 0.01,
                            7.5, 25, 12, 0.5])
    if c == 2:
        params0 = np.array([-0.3, -0.3, 1.0, 2,
                            -0.3, -0.3, -0.1, 3.0, 0.01,
                            7.5, 25, 12, 0.5])
        bounds = [(None,None), (None,None), (None,None), (None,None), 
              (None,None), (None,None), (None,None), (None,None), (None,None),
              (None,None), (None,None), (None,None), (0,None)]
    res = minimize(loss, params0, args=(data, time), method='L-BFGS-B', bounds=bounds)

    best_params = res.x
    print("Fitted global parameters:", best_params)
    cluster_params.append(best_params)
#%%
cluster_params_lf = []
for c in range(k):
    data = odtrips[cluster_euc[str(c)]]
    time = np.arange(data.shape[1])
    # Initial guess
    init_params = [0.5, 0.5, 1.0, 10.0]
    res = minimize(loss_logistic, init_params,
               args=(data, time), method='Nelder-Mead')

    best_params = res.x
    print("Fitted global parameters:", best_params)
    cluster_params_lf.append(best_params)
#%%
# Calculate calibration accuracy for ILRF
all_r2 = []
all_mape = []
for c in range(k):
    cluster_curves = []
    data = odtrips[cluster_euc[str(c)]]
    r2_scores = np.zeros(len(data))
    mape_scores = np.zeros(len(data))
    for i, _ in enumerate(data):
        y_t = data[i]
        mu, sigma = np.mean(y_t), np.std(y_t)
        y_p = resilience_curve(time, mu, sigma, cluster_params[c])
        # ---- R^2 for sample i ----
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
#%%
all_r2_lf = []
all_mape_lf = []
for c in range(k):
    cluster_curves = []
    data = odtrips[cluster_euc[str(c)]]
    r2_scores = np.zeros(len(data))
    mape_scores = np.zeros(len(data))
    for i, _ in enumerate(data):
        y_t = data[i]
        mu, sigma = np.mean(y_t), np.std(y_t)
        y_p = general_logistic(time, mu, sigma, cluster_params_lf[c])
        # ---- R^2 for sample i ----
        r2_scores[i] = np.sqrt((r_regression(y_p.reshape(-1, 1), y_t))**2)

        # ---- MAPE for sample i ----
        den = (np.abs(y_t) + np.abs(y_p)) / 2
        mask = den < 0.1
        smape = np.empty_like(den)
        smape[mask] = np.nan
        smape[~mask] = np.abs(y_t[~mask] - y_p[~mask]) / den[~mask]

        smape = np.nanmean(smape) * 100
        mape_scores[i] = smape

    all_r2_lf.append(r2_scores)
    all_mape_lf.append(mape_scores)
#%%
r2_sum = 0
r2_sum_lf = 0
mape_sum = 0
mape_sum_lf = 0
for c in range(k):
    # ILRF
    r2c = all_r2[c]
    finite_mean = np.mean(r2c[np.isfinite(r2c)])
    r2c[np.isinf(r2c)] = finite_mean
    r2_sum += r2c.sum()
    mape_sum += all_mape[c].sum()
    # Logistic
    r2c = all_r2_lf[c]
    finite_mean = np.mean(r2c[np.isfinite(r2c)])
    r2c[np.isinf(r2c)] = finite_mean
    r2_sum_lf += r2c.sum()
    mape_sum_lf += all_mape_lf[c].sum()
    
print("==============Euclidean + ILRF metrics===============")
print(f"Avg $R^2$: ", r2_sum/odtrips.shape[0])
print(f"Avg SMAPE: ", mape_sum/odtrips.shape[0])
print("==============Euclidean + logistic metrics===============")
print(f"Avg $R^2$: ", r2_sum_lf/odtrips.shape[0])
print(f"Avg SMAPE: ", mape_sum_lf/odtrips.shape[0])
# %%
# COVID
# ==============Euclidean + ILRF metrics===============
# Avg $R^2$:  0.9152133885393947
# Avg SMAPE:  29.193405444806114
# ==============Euclidean + logistic metrics===============
# Avg $R^2$:  0.8606186694839341
# Avg SMAPE:  49.46129476716272
# ==============Euclidean + ILRF metrics===============
# Avg $R^2$:  0.4156302486522335
# Avg SMAPE:  40.33996172652467
# ==============Euclidean + logistic metrics===============
# Avg $R^2$:  0.004418873521003547
# Avg SMAPE:  42.77005427109565