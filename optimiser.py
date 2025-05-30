import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from run_simulation import run_simulation
from pyDOE import lhs
from plotting import (
    plot_final_surface, plot_variance_surface,
    plot_voronoi, plot_acquisition_trace
)

# Hyperparameters
INIT_SAMPLES = 10
MAX_ITER = 5000
noise_std = 300.0

# Kalman Filter
class KalmanProfitFilter:
    def __init__(self):
        self.mu = {}
        self.var = {}

    def update(self, x, y_obs, noise=noise_std):
        x_key = tuple(x)
        if x_key not in self.mu:
            self.mu[x_key] = y_obs
            self.var[x_key] = noise**2
        else:
            prior_mu = self.mu[x_key]
            prior_var = self.var[x_key]
            kalman_gain = prior_var / (prior_var + noise**2)
            new_mu = prior_mu + kalman_gain * (y_obs - prior_mu)
            new_var = (1 - kalman_gain) * prior_var
            self.mu[x_key] = new_mu
            self.var[x_key] = new_var

    def get(self, x):
        x_key = tuple(x)
        return self.mu.get(x_key, 0), self.var.get(x_key, noise_std**2)

# Parameter space
price_range = np.linspace(1, 10.0, 100)
baristas_range = np.arange(1, 6)

def sample_params(n):
    # Generate LHS samples in 2D [baristas, price] ∈ [0,1]
    lhs_samples = lhs(2, samples=n, criterion='maximin')

    # Scale baristas from [0,1] to discrete values in baristas_range
    baristas_idx = np.floor(lhs_samples[:, 0] * len(baristas_range)).astype(int)
    baristas_idx = np.clip(baristas_idx, 0, len(baristas_range) - 1)
    baristas = baristas_range[baristas_idx]

    # Scale prices from [0,1] to continuous values in price_range range
    prices = lhs_samples[:, 1] * (price_range[-1] - price_range[0]) + price_range[0]

    return np.column_stack((baristas, prices))

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

def acquisition(mu, sigma, beta=5.0):
    return mu + beta * np.sqrt(sigma)

def main():
    data = []
    kf = KalmanProfitFilter()
    acquisition_trace = []

    samples = sample_params(INIT_SAMPLES)
    for p in samples:
        profit = run_simulation(p[1], p[0])
        kf.update(p, profit)
        data.append((*p, profit))

    for iter in range(MAX_ITER):
        df = pd.DataFrame(data, columns=['baristas', 'price', 'profit'])
        points = df[['baristas', 'price']].values
        profits = df['profit'].values
        grid = np.array([[b, p] for b in baristas_range for p in price_range])

        try:
            mu_pred = interpolate_profit(points, profits, grid)
        except:
            mu_pred = np.zeros(len(grid))

        known_variances = np.array([kf.get(pt)[1] for pt in points])
        try:
            sigma_pred = interpolate_profit(points, known_variances, grid)
        except:
            sigma_pred = np.full(len(grid), noise_std**2)

        scores = acquisition(mu_pred, sigma_pred)
        best_idx = np.argmax(scores)
        acquisition_trace.append(scores[best_idx])
        next_point = grid[best_idx]
        baristas, price = next_point

        profit = run_simulation(price, baristas)
        kf.update(next_point, profit)
        data.append((baristas, price, profit))

        if (iter + 1) % 100 == 0:
            print(f"[Iter {iter+1}] Tested (baristas={baristas}, price={price:.2f}) → Profit = {profit:.2f}")

    df_final = pd.DataFrame(data, columns=['baristas', 'price', 'profit'])
    df_final.to_csv("results.csv", index=False)
    print("Optimization finished. Results saved to results.csv")

    plot_variance_surface(grid, sigma_pred, baristas_range, price_range)
    plot_final_surface(df_final, baristas_range, price_range)
    plot_voronoi(df_final)
    plot_acquisition_trace(acquisition_trace)

if __name__ == "__main__":
    main()
