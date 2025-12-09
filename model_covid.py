#%%
import numpy as np
from scipy.optimize import minimize
import pickle

from src.clustering import DTWClustering
from src.resilience_model import resilience_curve, loss
from src.miscs import seed_everything

seed_everything(11)
#%%
# load data
odtrips = np.load('results/covid_demand_normalized.npy')
# perform clustering
np.random.seed(11)
model = DTWClustering(data=odtrips, k=4, max_iters=10)
model.train()
# %%
cluster_params = []
for c in range(4):
    data = odtrips[model.clusters[c]]
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
# Find the best-fitting time series
all_losses = []
for c in range(4):
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
f = open('results/results_covid2.pkl', 'wb')
pickle.dump([model.clusters, model.centroids, cluster_params, all_losses], f)
f.close()
# %%
