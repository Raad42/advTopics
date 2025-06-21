import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

def run_simulation(price, wage, noise_std=0.5):
    # Example behavior: peak around price=3, wage=25
    base_value = np.sin(price) * np.exp(-((wage - 25) ** 2) / 50) + np.cos(wage / 5)
    noise = np.random.normal(0, noise_std)  # Gaussian noise with std dev noise_std
    return base_value + noise

def averaged_objective(X, n_runs=5):
    """
    Evaluate the noisy simulator multiple times per input point and return the mean.
    X: array of shape (n_samples, 2)
    Returns: array of shape (n_samples,)
    """
    n_samples = X.shape[0]
    values = np.zeros((n_runs, n_samples))
    
    for run_idx in range(n_runs):
        for i in range(n_samples):
            values[run_idx, i] = run_simulation(X[i, 0], X[i, 1])
            
    return np.mean(values, axis=0)


def bayes_opt_ucb_continuous(
    objective_fn,
    bounds,
    n_init=10,
    n_iter=30,
    kappa=2.0,
    random_state=None
):
    rng = np.random.RandomState(random_state)

    # Initial random samples within bounds
    X_obs = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_init, 2))
    y_obs = objective_fn(X_obs)

    # Define kernel for GP
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) *
        Matern(length_scale=1.0, nu=2.5) +
        WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=random_state)

    # Create grid for acquisition function evaluation
    grid_size = 100
    p_lin = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    w_lin = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    P, W = np.meshgrid(p_lin, w_lin)
    X_grid = np.column_stack([P.ravel(), W.ravel()])

    ucb_values_over_time = []

    for i in range(n_iter):
        gp.fit(X_obs, y_obs)
        mu, std = gp.predict(X_grid, return_std=True)
        ucb = mu + kappa * std

        ucb_values_over_time.append(np.max(ucb))

        next_idx = np.argmax(ucb)
        x_next = X_grid[next_idx:next_idx + 1]
        y_next = objective_fn(x_next)

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

        print(f"Iter {i+1:2d}: price={x_next[0,0]:.3f}, wage={x_next[0,1]:.3f}, value={y_next[0]:.4f}")

    # Final fit after iterations
    gp.fit(X_obs, y_obs)
    best_idx = np.argmax(y_obs)
    best_point = X_obs[best_idx]
    best_value = y_obs[best_idx]
    print(f"\nBest found: price={best_point[0]:.3f}, wage={best_point[1]:.3f}, value={best_value:.4f}")

    return X_obs, y_obs, gp, P, W, mu, std, ucb_values_over_time


if __name__ == "__main__":
    bounds = np.array([
        [1.0, 10.0],   # Price range
        [10.0, 30.0]   # Wage range
    ])

    # Run Bayesian optimization with UCB acquisition
    X_obs, y_obs, gp, P, W, MU_flat, SIGMA_flat, ucb_trace = bayes_opt_ucb_continuous(
        objective_fn=lambda X: averaged_objective(X, n_runs=5),
        bounds=bounds,
        n_init=10,
        n_iter=30,
        kappa=2.0,
        random_state=42
    )

    grid_size = 100
    MU = MU_flat.reshape((grid_size, grid_size))
    VAR = (SIGMA_flat**2).reshape((grid_size, grid_size))

    # Plot predictive mean surface
    plt.figure(figsize=(6, 5))
    cs = plt.contourf(P, W, MU, levels=20)
    plt.colorbar(cs, label='Predictive Mean')
    plt.scatter(X_obs[:, 0], X_obs[:, 1], c='red', s=30, label='Samples')
    plt.title('GP Predictive Mean')
    plt.xlabel('Price')
    plt.ylabel('Wage')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gp_ucb_mean.png', dpi=300)
    plt.close()

    # Plot predictive variance surface
    plt.figure(figsize=(6, 5))
    cs2 = plt.contourf(P, W, VAR, levels=20)
    plt.colorbar(cs2, label='Predictive Variance')
    plt.scatter(X_obs[:, 0], X_obs[:, 1], c='red', s=30)
    plt.title('GP Predictive Variance')
    plt.xlabel('Price')
    plt.ylabel('Wage')
    plt.tight_layout()
    plt.savefig('gp_ucb_variance.png', dpi=300)
    plt.close()

    # Plot UCB acquisition max value over time
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(ucb_trace) + 1), ucb_trace, marker='o')
    plt.title("Max UCB Acquisition Value Over Time")
    plt.xlabel("Iteration")
    plt.ylabel("Max UCB Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ucb_over_time.png", dpi=300)
    plt.close()

    print("Saved 'gp_ucb_mean.png', 'gp_ucb_variance.png', and 'ucb_over_time.png'")
