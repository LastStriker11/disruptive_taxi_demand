#%%
import numpy as np
from scipy.stats import t
import pickle

from src.resilience_model import resilience_curve, loss
from src.clustering import DTWClustering
#%%
# ===============================
#  Numerical Jacobian
# ===============================
def numerical_jacobian(model_fun, paras, data, time, eps=1e-6):
    """
    model_fun(params) must return the FULL predicted vector (flattened).
    """
    k = len(paras)
    y0 = model_fun(paras)
    n = len(y0)

    J = np.zeros((n, k))

    for j in range(k):
        p_eps = paras.copy()
        p_eps[j] += eps
        y_plus = model_fun(p_eps)

        p_eps[j] -= 2*eps
        y_minus = model_fun(p_eps)

        J[:, j] = (y_plus - y_minus) / (2*eps)

    return J


# ===============================
#  FULL MODEL PREDICTION (flattened)
# ===============================
def full_prediction(params, data, time):
    preds = []
    for i in range(data.shape[0]):
        series = data[i]
        mu = np.mean(series)
        sigma = np.std(series)
        pred = resilience_curve(time, mu, sigma, params)
        preds.append(pred)
    return np.concatenate(preds)


# ===============================
#  FULL RAW RESIDUAL VECTOR
# ===============================
def full_residuals(params, data, time):
    """
    Returns residuals = y - y_hat, flattened.
    """
    res = []
    for i in range(data.shape[0]):
        series = data[i]
        mu = np.mean(series)
        sigma = np.std(series)
        pred = resilience_curve(time, mu, sigma, params)
        res.append(series - pred)
    return np.concatenate(res)


# ===============================
#  MAIN SIGNIFICANCE FUNCTION
# ===============================
def parameter_significance(paras, data, time):
    """
    paras: 1D fitted parameter vector
    data: (n_series, T)
    time: (T,)
    """
    # ---- 1) Recompute raw residuals ----
    residuals = full_residuals(paras, data, time)
    n = len(residuals)
    k = len(paras)

    # ---- 2) Residual variance ----
    sigma2 = np.sum(residuals**2) / (n - k)

    # ---- 3) Jacobian ----
    J = numerical_jacobian(lambda p: full_prediction(p, data, time),
                           paras, data, time)

    # ---- 4) Parameter covariance ----
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    print("Singular values:", S)
    # JTJ_inv = np.linalg.inv(J.T @ J)
    JTJ_inv = np.linalg.pinv(J.T @ J)
    cov = sigma2 * JTJ_inv

    # ---- 5) Standard errors ----
    se = np.sqrt(np.diag(cov))

    # ---- 6) t-statistics ----
    t_stats = paras / se

    # ---- 7) p-values ----
    df = n - k
    p_vals = 2 * (1 - t.cdf(np.abs(t_stats), df=df))

    return {
        "params": paras,
        "std_errors": se,
        "t_stats": t_stats,
        "p_values": p_vals,
        "cov_matrix": cov
    }
#%%
# ===============================
#  EXAMPLE CALL
# ===============================
# load data and model
odtrips = np.load('results/covid_demand_normalized.npy')
model = DTWClustering(odtrips, 4)
model.clusters, model.centroids, cluster_params, all_losses = pickle.load(open('results/results_covid.pkl', 'rb'))
time = range(len(model.centroids[0]))
#%%
for c in range(4):
    data = odtrips[model.clusters[c]]
    result = parameter_significance(cluster_params[c], data, time)
    print("p-values:", result["p_values"])
# %%
# "alpha_d", "beta_d", "k_d", "v_d",
#     "alpha_r", "beta_r", "k_r", "v_r", "m",
#     "t_d", "t_r", "t_s", "k_s"