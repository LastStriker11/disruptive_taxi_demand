import numpy as np
from scipy.optimize import minimize

def resilience_curve(t, mu, sigma, params):
    """
    Connected Z-shaped + flapped Z resilience curve.
    
    t : array of time steps
    mu, sigma : mean and std of the time series
    params : list or array of global parameters
        [alpha_d, beta_d, k_d, v_d,
         alpha_r, beta_r, k_r, v_r, m,
         t_d, t_r, t_s, k_s]
    """
    (alpha_d, beta_d, k_d, v_d,
     alpha_r, beta_r, k_r, v_r, m,
     t_d, t_r, t_s, k_s) = params
    
    # Phase amplitudes
    A_d = alpha_d * sigma + beta_d * mu
    A_r = alpha_r * sigma + beta_r * mu
    
    # Disruption phase
    D = A_d / (1 + np.exp(-k_d * (t - t_d)))**v_d + mu - A_d/2
    
    # Recovery phase
    R = A_r / (1 + np.exp(-k_r * (t - t_r)))**v_r + mu - A_r/2 + m * (t - t_r)
    
    # Smooth transition
    S = 1 / (1 + np.exp(-k_s * (t - t_s)))
    
    return (1 - S) * D + S * R

def resilience_curve_sa(t, mu_Ad, sigma_Ad, mu_Ar, sigma_Ar, params):
    """
    Connected Z-shaped + flapped Z resilience curve.
    
    t : array of time steps
    mu, sigma : mean and std of the time series
    params : list or array of global parameters
        [alpha_d, beta_d, k_d, v_d,
         alpha_r, beta_r, k_r, v_r, m,
         t_d, t_r, t_s, k_s]
    """
    (alpha_d, beta_d, k_d, v_d,
     alpha_r, beta_r, k_r, v_r, m,
     t_d, t_r, t_s, k_s) = params
    
    # Phase amplitudes
    A_d = alpha_d * sigma_Ad + beta_d * mu_Ad
    A_r = alpha_r * sigma_Ar + beta_r * mu_Ar
    
    # Disruption phase
    D = A_d / (1 + np.exp(-k_d * (t - t_d)))**v_d + mu_Ad - A_d/2
    
    # Recovery phase
    R = A_r / (1 + np.exp(-k_r * (t - t_r)))**v_r + mu_Ar - A_r/2 + m * (t - t_r)
    
    # Smooth transition
    S = 1 / (1 + np.exp(-k_s * (t - t_s)))
    
    return (1 - S) * D + S * R

def loss(params, data, time):
    n_series = data.shape[0]
    loss_val = 0.0
    for i in range(n_series):
        series = data[i]
        mu = np.mean(series)
        sigma = np.std(series)
        pred = resilience_curve(time, mu, sigma, params)
        loss_val += np.sum((series - pred)**2)
    return loss_val

def general_logistic(t, mu, sigma, params):
    """
    General logistic curve with the same input structure as resilience_curve.

    t : array of time steps
    mu, sigma : mean and std of the time series
    params : list or array of logistic parameters
        [alpha, beta, k, t0]
    """
    alpha, beta, k, t0 = params

    # amplitude based on mu and sigma (parallel to your model)
    A = alpha * sigma + beta * mu  

    # general logistic function
    L = A / (1 + np.exp(-k * (t - t0))) + mu - A/2

    return L

def loss_logistic(params, data, time):
    n_series = data.shape[0]
    loss_val = 0.0
    for i in range(n_series):
        series = data[i]
        mu = np.mean(series)
        sigma = np.std(series)
        pred = general_logistic(time, mu, sigma, params)
        loss_val += np.sum((series - pred) ** 2)
    return loss_val

import numpy as np
from scipy import integrate

def area_below_line(x, y, line_y):
    """
    Calculate the area between the curve and the horizontal line,
    only where the curve is below the line.

    Parameters:
        x      : array of x values
        y      : array of y values (the curve)
        line_y : y-value of the horizontal line

    Returns:
        Area of the region where curve < line_y
    """
    x = np.array(x)
    y = np.array(y)
    diff = line_y - y  # positive where curve is below the line

    # If curve is entirely above the line, no area
    if np.all(diff <= 0):
        return 0.0

    # If curve is entirely below the line, integrate everything
    if np.all(diff >= 0):
        return integrate.trapezoid(diff, x)

    # Otherwise, find intersections and insert them into the grid
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    x_intersections = []
    for idx in sign_changes:
        x0, x1 = x[idx], x[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
        x_intersections.append(x_cross)

    # Refine grid with intersection points
    x_full = np.sort(np.unique(np.concatenate([x, x_intersections])))
    diff_full = line_y - np.interp(x_full, x, y)

    # Only integrate the parts where curve is below the line
    diff_clipped = np.clip(diff_full, 0, None)
    return integrate.trapezoid(diff_clipped, x_full)
