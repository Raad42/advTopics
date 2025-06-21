# optimisers/optimiser.py
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from scipy.stats.qmc import LatinHypercube, scale
from run_simulation import run_simulation  # your noisy simulator function

_sim_time_total = 0.0  # Global time accumulator

def timed_run_simulation(p, w):
    global _sim_time_total
    t0 = time.time()
    out = run_simulation(p, w)
    _sim_time_total += (time.time() - t0)
    return out

def averaged_objective(X, n_runs=1):
    n_samples = X.shape[0]
    values = np.zeros((n_runs, n_samples))
    for run_idx in range(n_runs):
        for i in range(n_samples):
            p, w = X[i, 0], X[i, 1]
            values[run_idx, i] = timed_run_simulation(p, w)
    return np.mean(values, axis=0)

def run_bayes_opt(bounds, n_init=5, n_iter=100, kappa=2.0, random_state=42):
    """
    Bayesian optimization with UCB acquisition.
    Returns:
      mean_profit     : re-evaluated best observed profit (50x avg)
      iter_to_best    : iteration where best profit was first selected
      time_to_best    : wall-clock time to reach best
      iter_to_95pct   : iteration where 95% of best was first reached
      time_to_95pct   : wall-clock time to reach 95% of best
      best_iter       : index of best overall observation
      total_time      : total wall-clock time
      sim_time_total  : accumulated simulation time
      comp_time_total : total_time - sim_time_total
    """
    global _sim_time_total
    _sim_time_total = 0.0

    # Initial Latin Hypercube Sampling
    sampler = LatinHypercube(d=2, seed=random_state)
    lhs = sampler.random(n_init)
    X_obs = scale(lhs, bounds[:, 0], bounds[:, 1])
    y_obs = averaged_objective(X_obs, n_runs=1)

    # GP model setup
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        Matern(length_scale=1.0, nu=2.5) +
        WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)

    # Acquisition search grid
    grid_size = 100
    p_lin = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    w_lin = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    P, W = np.meshgrid(p_lin, w_lin)
    X_grid = np.column_stack([P.ravel(), W.ravel()])

    sim_trace = []
    total_trace = []

    wall_start = time.time()
    sim_start_ref = _sim_time_total

    for it in range(1, n_iter + 1):
        loop_start_wall = time.time()
        loop_start_sim = _sim_time_total

        gp.fit(X_obs, y_obs)
        mu, std = gp.predict(X_grid, return_std=True)
        ucb = mu + kappa * std
        idx = np.argmax(ucb)
        x_next = X_grid[idx:idx+1]
        y_next = averaged_objective(x_next, n_runs=1)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

        loop_end_wall = time.time()
        loop_end_sim = _sim_time_total

        sim_elapsed = loop_end_sim - loop_start_sim
        wall_elapsed = loop_end_wall - loop_start_wall

        sim_trace.append(loop_end_sim - sim_start_ref)
        total_trace.append((loop_end_sim - sim_start_ref) + (wall_elapsed - sim_elapsed))

    total_time = time.time() - wall_start

    # Best point
    best_idx = np.argmax(y_obs)
    best_point = X_obs[best_idx]

    if best_idx < n_init:
        iter_to_best = 0
        time_to_best = 0.0
    else:
        iter_to_best = best_idx - n_init + 1
        time_to_best = total_trace[iter_to_best - 1]

    # Re-evaluate best point 50 times
    reeval_runs = 50
    best_profit_samples = [timed_run_simulation(*best_point) for _ in range(reeval_runs)]
    mean_profit = np.mean(best_profit_samples)

    # Compute 95% threshold stats
    best_so_far = [np.max(y_obs[:n_init + i]) for i in range(n_iter + 1)]
    threshold_95 = 0.95 * mean_profit
    iter_95 = None
    time_95 = None
    for i, val in enumerate(best_so_far[n_init:]):
        if val >= threshold_95:
            iter_95 = i + 1
            time_95 = total_trace[i]
            break

    return {
        'mean_profit': mean_profit,
        'iter_to_best': iter_to_best,
        'time_to_best': time_to_best,
        'iter_to_95pct': iter_95,
        'time_to_95pct': time_95,
        'best_iter': best_idx + 1,
        'total_time': total_time,
        'sim_time_total': _sim_time_total,
        'comp_time_total': total_time - _sim_time_total
    }
