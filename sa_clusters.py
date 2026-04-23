#%%
import numpy as np
from scipy.optimize import minimize
import pickle
import matplotlib.pyplot as plt
import time
from sklearn.metrics import silhouette_score

from src.clustering import DTWClustering
from src.resilience_model import loss
from src.analysis import compute_r2_smape_per_sample, silhouette_score_dtw_memory_efficient
from src.miscs import seed_everything

seed_everything(11)
#%%
# load data
odtrips = np.load('results/covid_demand_normalized.npy')
# perform clustering
np.random.seed(11)
#%%
scores = []
compute_time = []
tic = time.time()
for num_clusters in [2,3,4,5,6]:
    model = DTWClustering(odtrips, k=num_clusters, max_iters=100)
    model.train()
    toc = time.time()
    compute_time.append(toc - tic)

    n_samples = sum(len(indices) for indices in model.clusters.values())
    cluster_arr = np.zeros(n_samples, dtype=int)

    for label, indices in model.clusters.items():
        cluster_arr[indices] = int(label)
    score = silhouette_score_dtw_memory_efficient(odtrips, cluster_arr)
    scores.append(score)

    tic = time.time()
#%%
r2_all = []
mape_all = []
compute_time = []
tic = time.time()
for num_clusters in [2,3,4,5,6]:
    model = DTWClustering(data=odtrips, k=num_clusters, max_iters=10)
    model.train()
    toc = time.time()
    compute_time.append(toc - tic)
    cluster_params = []
    for c in range(num_clusters):
        data = odtrips[model.clusters[c]]
        t_range = np.arange(data.shape[1])
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
        res = minimize(loss, params0, args=(data, t_range), method='L-BFGS-B', bounds=bounds)

        best_params = res.x
        cluster_params.append(best_params)
    
    r2_per_sample, mape_per_sample = compute_r2_smape_per_sample(odtrips, model.centroids, model.clusters)
    r2_all.append(r2_per_sample)
    mape_all.append(mape_per_sample)
    
    print(f"Number of clusters: {num_clusters}, R2: {r2_per_sample.mean():.4f} ± {r2_per_sample.std():.4f}, MAPE: {mape_per_sample.mean():.4f} ± {mape_per_sample.std():.4f}, Time: {(toc-tic)/60:.2f} mins")
    tic = time.time()
# %%
f = open('results/results_sa_clusters.pkl', 'wb')
pickle.dump([r2_all, mape_all, compute_time], f)
f.close()
# %%
for num_clusters in [2,3,4,5,6]:
    print(f"Number of clusters: {num_clusters}")
    print(f"R2: {r2_all[num_clusters-2].mean():.4f} ± {r2_all[num_clusters-2].std():.4f}")
    print(f"MAPE: {mape_all[num_clusters-2].mean():.4f} ± {mape_all[num_clusters-2].std():.4f}")
    print(f"Time: {compute_time[num_clusters-2]:.3f} s")
# %%
fig, ax = plt.subplots(figsize=(6,4))
ax.bar([2,3,4,5,6], [r2.mean() for r2 in r2_all], marker='o')
ax.plot
# %%
x = np.arange(2, 7)  # cluster numbers
width = 0.35  # bar width
# -------------------------------------
fig, ax1 = plt.subplots(figsize=(5,2.7))
ax1.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax1.spines[['bottom','top','left']].set_linewidth(1.5)
ax1.spines["right"].set_visible(False)

# --- Left axis (Abs. Pearson) ---
ax1.set_xlabel(r"Number of clusters ($k$)", fontsize=12)
ax1.set_ylabel(r"Absolute Pearson correlation", color="#001BB7")
bars_r2 = ax1.bar(x - width/2, [r2.mean() for r2 in r2_all], width, yerr=[r2.std() for r2 in r2_all], 
                  capsize=5, label=r"Abs. Pearson",
                  edgecolor="#001BB7", facecolor="#AAC4F5", ecolor="#001BB7")
ax1.tick_params(axis="y", colors="#001BB7")
ax1.spines["left"].set_color("#001BB7")
ax1.set_ylim([0.75, 1.1])
ax1.set_yticks([0.8,0.9,1])

# --- Right axis (MAPE) ---
ax2 = ax1.twinx()
ax2.set_ylabel("Computation time (s)", color="#A72703")
bars_mape = ax2.bar(x + width/2, compute_time, width, 
                    capsize=5, label="Computation time", edgecolor="#A72703", facecolor="#FFF2EF", ecolor="#A72703")
ax2.tick_params(direction='in', top=True, right=True, which='both', width=1.5)
ax2.spines[['bottom','right']].set_linewidth(1.5)
ax2.spines[['top','left']].set_visible(False)
ax2.tick_params(axis="y", colors="#A72703")
ax2.spines["right"].set_color("#A72703")
ax2.set_ylim([0,10])

# X axis labels
ax1.set_xticks(x)
ax1.set_xticklabels([f"$k$={i}" for i in x])
handles = [bars_r2, bars_mape]
labels = [r"Abs. Pearson", "Commputation time"]
ax1.legend(handles, labels, ncols=2, loc="upper right")

fig.savefig("figures/sa_cluster_k.pdf", bbox_inches="tight")
# %%
