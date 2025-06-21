import numpy as np
from scipy.spatial import Delaunay, KDTree

# -------------------------------
# 0. Deterministic objective
# -------------------------------
def run_simulation(price: float, wage: float) -> float:
    return np.sin(price / 2) * np.exp(-((wage - 20) ** 2) / 50)

# -------------------------------
# 1. Latin Hypercube Sampling (LHS)
# -------------------------------
BOUNDS = [(1.0, 2.5), (18.0, 22.0)]

def lhs_samples_2d(n, bounds):
    dim = len(bounds)
    cut = np.linspace(0, 1, n + 1)
    u = np.random.rand(n, dim)
    L = np.zeros((n, dim))
    for j in range(dim):
        perm = np.random.permutation(n)
        L[:, j] = cut[:-1] + u[:, j] * (1.0 / n)
        L[:, j] = L[perm, j]
    return np.array([bounds[j][0] + L[:, j] * (bounds[j][1] - bounds[j][0]) for j in range(dim)]).T

def initialize_with_lhs(n_init=10, R_noise=1e-3):
    hull_pts = [[BOUNDS[0][0], BOUNDS[1][0]], [BOUNDS[0][0], BOUNDS[1][1]],
                [BOUNDS[0][1], BOUNDS[1][0]], [BOUNDS[0][1], BOUNDS[1][1]]]

    nodes, profits = [], []
    best_obs, best_params = -np.inf, (None, None)

    for price, wage in hull_pts:
        y = run_simulation(price, wage)
        nodes.append([price, wage])
        profits.append(y)
        if y > best_obs:
            best_obs, best_params = y, (price, wage)

    n_lhs = max(0, n_init - len(hull_pts))
    lhs = lhs_samples_2d(n_lhs, BOUNDS) if n_lhs > 0 else np.empty((0, 2))
    for price, wage in lhs:
        y = run_simulation(price, wage)
        nodes.append([price, wage])
        profits.append(y)
        if y > best_obs:
            best_obs, best_params = y, (price, wage)

    mu = np.array(profits)
    P = 1e4 * np.eye(len(nodes))
    return nodes, mu, P, best_obs, best_params, R_noise

# -------------------------------
# 2. Interpolation logic
# -------------------------------
def barycentric_coordinates(tri_pts, x):
    T = np.vstack([tri_pts.T, np.ones(3)])
    v = np.append(x, 1.0)
    return np.linalg.solve(T, v)

def fallback_mean_sigma(x, nodes, mu, P, k=3, alpha=1.0):
    tree = KDTree(nodes)
    dists, idxs = tree.query(x, k=k)
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum()
    mu_interp = np.dot(weights, mu[idxs])
    prior_var = np.dot(weights**2, P[idxs, idxs])
    sigma = np.sqrt(prior_var + alpha * np.max(dists)**2)
    return mu_interp, sigma

def interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha=1.0):
    simplex = delaunay.find_simplex(x.reshape(1, -1))[0]
    if simplex < 0:
        return fallback_mean_sigma(x, nodes, mu, P, alpha=alpha)
    
    verts = delaunay.simplices[simplex]
    pts = np.array(nodes)[verts]
    lambdas = barycentric_coordinates(pts, x)
    mu_interp = np.dot(lambdas, mu[verts])
    prior_var = np.dot(lambdas**2, P[verts, verts])
    dists = np.linalg.norm(x - pts, axis=1)
    sigma = np.sqrt(prior_var + alpha * np.max(dists)**2)
    return mu_interp, sigma

# -------------------------------
# 3. Kalman update
# -------------------------------
def one_iteration(nodes, mu, P, next_point, best_obs, best_params, R_noise, alpha=1.0):
    x_new = np.asarray(next_point)
    tree = KDTree(nodes)
    _, idx = tree.query(x_new, k=1)
    if np.linalg.norm(np.asarray(nodes[idx]) - x_new) < 1e-6:
        y_new = run_simulation(*x_new)
        H = np.zeros((1, len(nodes)))
        H[0, idx] = 1.0
        S = H @ P @ H.T + R_noise
        K = (P @ H.T) / S
        mu_post = mu + (K.flatten() * (y_new - H @ mu))
        P_post = P - (K @ H @ P)
        if y_new > best_obs:
            best_obs = y_new
            best_params = tuple(x_new)
        return nodes, mu_post, P_post, best_obs, best_params

    if len(nodes) < 3:
        return nodes, mu, P, best_obs, best_params

    delaunay = Delaunay(np.asarray(nodes))
    mu0, sigma0 = interpolate_mean_and_sigma(x_new, nodes, mu, P, delaunay, alpha)
    N = len(nodes)
    cov_old = np.zeros(N)

    simplex = delaunay.find_simplex(x_new.reshape(1, -1))[0]
    if simplex >= 0:
        verts = delaunay.simplices[simplex]
        lambdas = barycentric_coordinates(np.asarray(nodes)[verts], x_new)
        for jj, ii in enumerate(verts):
            cov_old[ii] = lambdas[jj] * P[ii, ii]
    else:
        dists, idxs = tree.query(x_new, k=3)
        weights = 1.0 / (dists + 1e-6); weights /= weights.sum()
        for jj, ii in enumerate(np.atleast_1d(idxs)):
            cov_old[ii] = weights[jj] * P[ii, ii]

    mu_aug = np.append(mu, mu0)
    P_aug = np.zeros((N+1, N+1))
    P_aug[:N, :N] = P
    P_aug[:N, N] = cov_old
    P_aug[N, :N] = cov_old
    P_aug[N, N] = sigma0**2

    y_new = run_simulation(*x_new)
    H = np.zeros((1, N+1)); H[0, N] = 1.0
    S = H @ P_aug @ H.T + R_noise
    K = (P_aug @ H.T) / S
    mu_post = mu_aug + (K.flatten() * (y_new - H @ mu_aug))
    P_post = P_aug - (K @ H @ P_aug)

    nodes.append(x_new.tolist())
    if y_new > best_obs:
        best_obs, best_params = y_new, tuple(x_new)
    return nodes, mu_post, P_post, best_obs, best_params

# -------------------------------
# 4. UCB Proposal
# -------------------------------
def propose_next_ucb(nodes, mu, P, delaunay,
                     bounds=BOUNDS, M=1000, kappa=2.0, alpha=1.0):
    best_ucb, best_x = -np.inf, None
    for _ in range(M):
        x = np.array([np.random.uniform(low, high) for (low, high) in bounds])
        mu_val, sigma_val = interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha)
        ucb = mu_val + kappa * sigma_val
        if ucb > best_ucb:
            best_ucb, best_x = ucb, x.tolist()
    return best_x, best_ucb

# -------------------------------
# 5. Main Optimization Loop
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def plot_acquisition_trace(trace, filename="trace.png"):
        plt.figure(figsize=(8, 4))
        plt.plot(trace, label="UCB Trace")
        plt.xlabel("Iteration")
        plt.ylabel("UCB Acquisition Value")
        plt.title("Acquisition Function Trace")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    n_initial = 10
    R_noise_est = 1e-3
    kappa = 2.0
    alpha = 1.0
    n_iterations = 200
    M = 1000

    nodes, mu, P, best_obs, best_params, R_noise = initialize_with_lhs(n_init=n_initial, R_noise=R_noise_est)
    print("Initial best:", best_obs, "at", best_params)

    acq_trace = []

    for it in range(1, n_iterations + 1):
        delaunay = Delaunay(np.asarray(nodes))
        if np.random.rand() < 0.1:
            next_pt = [np.random.uniform(*BOUNDS[0]), np.random.uniform(*BOUNDS[1])]
            ucb_val = None
        else:
            next_pt, ucb_val = propose_next_ucb(nodes, mu, P, delaunay, BOUNDS, M=M, kappa=kappa, alpha=alpha)
        if ucb_val is not None:
            acq_trace.append(ucb_val)

        print(f"[Iter {it}] Next point: {next_pt}")
        nodes, mu, P, best_obs, best_params = one_iteration(nodes, mu, P, next_pt, best_obs, best_params, R_noise, alpha)

        if it % 10 == 0:
            print(f"Iter {it}: best observed = {best_obs:.6f} at {best_params}")

    print("=== Optimization complete ===")
    print(f"Best observed value: {best_obs:.6f} at {best_params}")
    plot_acquisition_trace(acq_trace, filename="final_acquisition_trace.png")
    print("Saved acquisition trace.")
