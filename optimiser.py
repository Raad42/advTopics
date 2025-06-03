import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from run_simulation import run_simulation
from pyDOE import lhs
from plotting import (
    plot_final_surface, plot_variance_surface,
    plot_voronoi, plot_acquisition_trace
)

# Hyperparameters
INIT_SAMPLES = 10
MAX_ITER = 2000
noise_std = 300.0
lengthscale = 1.0   # for RBF kernel
signal_var = 1.0    # prior signal variance

# Parameter space (grid)
baristas_range = np.arange(1, 6)
price_range    = np.linspace(1, 10.0, 100)
grid = np.array([[b, p] for b in baristas_range for p in price_range])
n_grid = len(grid)

# Vector Kalman Filter for variance only
class VectorKalmanVar:
    def __init__(self, grid, lengthscale, signal_var, noise_var):
        # Compute prior covariance K0 using RBF kernel
        dists = cdist(grid, grid, 'euclidean')
        K0 = signal_var * np.exp(-0.5 * (dists / lengthscale)**2)
        # Initial covariance: prior + measurement noise on diagonal
        self.P = K0 + noise_var * np.eye(len(grid))
        self.noise_var = noise_var

    def update(self, idx):
        # Kalman variance-only update for measurement at idx
        Pii = self.P[idx, idx]
        K   = self.P[:, idx] / (Pii + self.noise_var)
        # Covariance update: P <- P - K * P[idx, :]
        self.P -= np.outer(K, self.P[idx, :])

    def predict_var(self):
        # Return variance at all grid points
        return np.diag(self.P).copy()

# Delaunay interpolation for means
def interpolate_profit(points, values, queries):
    tri = Delaunay(points)
    simplex = tri.find_simplex(queries)
    valid = simplex >= 0
    X = tri.transform[simplex[valid], :2]
    Y = queries[valid] - tri.transform[simplex[valid], 2]
    bary = np.einsum('ijk,ik->ij', X, Y)
    bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]
    verts = tri.simplices[simplex[valid]]
    interpolated = np.zeros(len(queries))
    interpolated[valid] = np.einsum('ij,ij->i', values[verts], bary_coords)
    return interpolated

# Acquisition function (UCB)
def acquisition(mu, sigma, beta=2.0):
    return mu + beta * np.sqrt(sigma)

# Latin Hypercube sampling over grid indices
def sample_grid_indices(n):
    lhs_samples = lhs(2, samples=n, criterion='maximin')
    b_idx = np.floor(lhs_samples[:, 0] * len(baristas_range)).astype(int)
    p_idx = np.floor(lhs_samples[:, 1] * len(price_range)).astype(int)
    b_idx = np.clip(b_idx, 0, len(baristas_range)-1)
    p_idx = np.clip(p_idx, 0, len(price_range)-1)
    return b_idx * len(price_range) + p_idx

# Main optimization loop
def main():
    # Initialize variance filter
    vkf = VectorKalmanVar(grid, lengthscale, signal_var, noise_std**2)
    data = []       # list of (idx, profit)
    trace = []

    # Initial sampling
    init_idx = sample_grid_indices(INIT_SAMPLES)
    for idx in init_idx:
        b, p = grid[idx]
        profit = run_simulation(p, b)
        # record data and update variance
        data.append((idx, profit))
        vkf.update(idx)

    # Iterative optimization
    for i in range(MAX_ITER):
        # Retrieve sampled points for mean interpolation
        points = np.array([grid[idx] for idx, _ in data])
        profits = np.array([profit for _, profit in data])
        # Mean prediction via Delaunay
        mu_pred = interpolate_profit(points, profits, grid)
        # Variance prediction via Kalman
        sigma_pred = vkf.predict_var()

        # Acquisition
        scores = acquisition(mu_pred, sigma_pred)
        best_idx = np.argmax(scores)
        trace.append(scores[best_idx])

        # Evaluate and update
        b, p = grid[best_idx]
        profit = run_simulation(p, b)
        data.append((best_idx, profit))
        vkf.update(best_idx)

        if (i+1) % 100 == 0:
            print(f"[Iter {i+1}] idx={best_idx} (baristas={b}, price={p:.2f}) -> profit={profit:.2f}")

    # Save results
    df = pd.DataFrame([ (grid[idx,0], grid[idx,1], prof) for idx, prof in data ],
                      columns=['baristas','price','profit'])
    df.to_csv('results.csv', index=False)
    print('Done. Results in results.csv')

    # Final predictions
    mu_final = interpolate_profit(points, profits, grid)
    var_final = vkf.predict_var()

    # Plots
    plot_final_surface(grid, mu_final, baristas_range, price_range)
    plot_variance_surface(grid, var_final, baristas_range, price_range)
    plot_voronoi(df)
    plot_acquisition_trace(trace)

if __name__ == '__main__':
    main()
