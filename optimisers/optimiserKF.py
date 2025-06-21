import numpy as np
from scipy.spatial import Delaunay, KDTree
from run_simulation import run_simulation

from plots.plotting import (
    plot_final_surface,
    plot_variance_surface,
    plot_voronoi as plot_voronoi_new,
    plot_acquisition_trace
)

# ----------------------------------------------
# 1. Latin Hypercube Sampling (LHS) for 2D
# ----------------------------------------------
def lhs_samples_2d(n, bounds):
    dim = 2
    cut = np.linspace(0, 1, n + 1)
    u = np.random.rand(n, dim)
    a = np.zeros((n, dim))
    for j in range(dim):
        perm = np.random.permutation(n)
        a[:, j] = cut[perm]
    L = a + u * (1.0 / n)
    samples = np.zeros_like(L)
    for j in range(dim):
        lo, hi = bounds[j]
        samples[:, j] = lo + L[:, j] * (hi - lo)
    return samples

def initialize_with_lhs(n_init=10, R_noise=1.0):
    bounds = [(1.0, 10.0), (15.0, 30.0)]  # price, wage
    hull_pts = [[1.0, 15.0], [1.0, 30.0], [10.0, 15.0], [10.0, 30.0]]

    nodes, profits = [], []
    best_obs = -np.inf
    best_params = (None, None)

    for price, wage in hull_pts:
        y = run_simulation(price, wage)
        nodes.append([price, wage])
        profits.append(y)
        if y > best_obs:
            best_obs, best_params = y, (price, wage)

    n_lhs = max(0, n_init - len(hull_pts))
    cont = lhs_samples_2d(n_lhs, bounds) if n_lhs > 0 else np.empty((0,2))

    for price, wage in cont:
        y = run_simulation(price, wage)
        nodes.append([price, wage])
        profits.append(y)
        if y > best_obs:
            best_obs, best_params = y, (price, wage)

    mu = np.array(profits)
    σ0 = 1e4
    P = σ0 * np.eye(len(nodes))
    return nodes, mu, P, best_obs, best_params, R_noise

# ----------------------------------------------
# 2. Interpolation helpers & fallback
# ----------------------------------------------
def barycentric_coordinates(tri_pts, x):
    T = np.vstack([tri_pts.T, np.ones(3)])
    v = np.append(x, 1.0)
    return np.linalg.solve(T, v)

def fallback_mean_sigma(x, nodes, mu, P, k=3, alpha=1.0):
    tree = KDTree(nodes)
    dists, idxs = tree.query(x, k=k)
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)

    weights = 1.0 / (dists + 1e-6)
    weights /= np.sum(weights)

    mu_interp = sum(weights[i] * mu[idxs[i]] for i in range(len(idxs)))
    prior_var = sum((weights[i]**2) * P[idxs[i], idxs[i]] for i in range(len(idxs)))
    sigma = np.sqrt(prior_var + alpha * (np.max(dists)**2))
    return mu_interp, sigma

def interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha=1.0):
    idx = int(delaunay.find_simplex(x.reshape(1, -1))[0])
    if idx < 0:
        return fallback_mean_sigma(x, nodes, mu, P, k=3, alpha=alpha)

    verts = delaunay.simplices[idx]
    pts = np.asarray(nodes)[verts]
    λ = barycentric_coordinates(pts, x)

    mu_interp = sum(λ[i] * mu[verts[i]] for i in range(3))
    prior_var = sum((λ[i]**2) * P[verts[i], verts[i]] for i in range(3))
    dists = [np.linalg.norm(x - pts[i]) for i in range(3)]
    sigma = np.sqrt(prior_var + alpha * (max(dists)**2))

    return mu_interp, sigma

# ----------------------------------------------
# 3. Kalman update
# ----------------------------------------------
def one_iteration(nodes, mu, P, next_point, best_obs, best_params, R_noise, alpha=2.0):
    x_new = np.asarray(next_point)
    for i, pt in enumerate(nodes):
        if np.allclose(pt, x_new, atol=1e-6):
            y_new = run_simulation(x_new[0], x_new[1])
            H = np.zeros((1, len(nodes))); H[0, i] = 1.0
            S = H @ P @ H.T + R_noise
            K = (P @ H.T) / S
            mu_post = mu + (K.flatten() * (y_new - H @ mu))
            P_post = P - (K @ H @ P)
            if y_new > best_obs:
                best_obs, best_params = y_new, (x_new[0], x_new[1])
            return nodes, mu_post, P_post, best_obs, best_params

    if len(nodes) < 3:
        return nodes, mu, P, best_obs, best_params

    delaunay = Delaunay(np.asarray(nodes))
    mu0, sigma0 = interpolate_mean_and_sigma(x_new, nodes, mu, P, delaunay, alpha)
    N = len(nodes)

    cov_old = np.zeros(N)
    tri_idx = int(delaunay.find_simplex(x_new.reshape(1, -1))[0])
    if tri_idx >= 0:
        verts = delaunay.simplices[tri_idx]
        λ = barycentric_coordinates(np.asarray(nodes)[verts], x_new)
        for jj, ii in enumerate(verts):
            cov_old[ii] = λ[jj] * P[ii, ii]
    else:
        tree = KDTree(nodes)
        dists, idxs = tree.query(x_new, k=3)
        weights = 1.0 / (dists + 1e-6)
        weights /= np.sum(weights)
        for jj, ii in enumerate(np.atleast_1d(idxs)):
            cov_old[ii] = weights[jj] * P[ii, ii]

    mu_aug = np.concatenate([mu, [mu0]])
    P_aug = np.zeros((N+1, N+1))
    P_aug[:N, :N] = P
    P_aug[:N, N] = cov_old
    P_aug[N, :N] = cov_old
    P_aug[N, N] = sigma0**2

    y_new = run_simulation(x_new[0], x_new[1])
    H = np.zeros((1, N+1)); H[0, N] = 1.0
    S = H @ P_aug @ H.T + R_noise
    K = (P_aug @ H.T) / S
    mu_post = mu_aug + (K.flatten() * (y_new - H @ mu_aug))
    P_post = P_aug - (K @ H @ P_aug)

    nodes_new = nodes + [[x_new[0], x_new[1]]]
    if y_new > best_obs:
        best_obs, best_params = y_new, (x_new[0], x_new[1])
    return nodes_new, mu_post, P_post, best_obs, best_params

# ----------------------------------------------
# 4. UCB proposal
# ----------------------------------------------
def propose_next_ucb_price_wage(nodes, mu, P, delaunay,
                                M=4000, kappa=10.0, alpha=3.0):
    best_ucb = -np.inf
    best_cand = None
    print("\n--- UCB Evaluation for Proposed Points ---")
    prices = np.random.uniform(1, 10, M)
    wages = np.random.uniform(15, 30, M)
    for price, wage in zip(prices, wages):
        x = np.array([price, wage])
        μ_val, σ_val = interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha)
        ucb = μ_val + kappa * σ_val

        if ucb > best_ucb:
            best_ucb, best_cand = ucb, [price, wage]
    print("--- End of UCB Evaluation ---\n")
    return best_cand, best_ucb

# ----------------------------------------------
# 5. Main optimization loop
# ----------------------------------------------
if __name__ == "__main__":
    n_initial, R_noise_est = 10, 450**2
    nodes, mu, P, best_obs, best_params, R_noise = initialize_with_lhs(
        n_init=n_initial, R_noise=R_noise_est
    )
    print("Initial best:", best_obs, "at", best_params)

    acq_trace = []
    n_iterations, kappa, alpha = 200, 10.0, 3.0

    for iteration in range(1, n_iterations + 1):
        delaunay = Delaunay(np.asarray(nodes))

        if np.random.rand() < 0.1:
            next_pt = [np.random.uniform(1, 10), np.random.uniform(15, 30)]
            μ_val, σ_val, ucb_val = None, None, None
        else:
            next_pt, ucb_val = propose_next_ucb_price_wage(
                nodes, mu, P, delaunay, M=400, kappa=kappa, alpha=alpha
            )
            μ_val, σ_val = interpolate_mean_and_sigma(
                np.array(next_pt), nodes, mu, P, delaunay, alpha=alpha
            )

        if ucb_val is not None:
            acq_trace.append(ucb_val)

        print(f"[Iter {iteration}] Next point: {next_pt}")
        if μ_val is not None and σ_val is not None:
            print(f"  UCB: {ucb_val:.3f}, Posterior μ: {μ_val:.3f}, σ²: {σ_val:.3f}")

        nodes, mu, P, best_obs, best_params = one_iteration(
            nodes, mu, P, next_pt, best_obs, best_params, R_noise, alpha=alpha
        )

        if iteration % 10 == 0:
            print(f"Iter {iteration}: best observed = {best_obs:.3f} at {best_params}")

        if iteration in [50, 100, 200]:
            print("r")
            #print(f"--- Generating plots at iter {iteration} ---")
            #prices = np.linspace(1, 10, 50)
           # wages = np.linspace(15, 30, 50)
            #grid = np.array([[p, w] for p in prices for w in wages])

            #μg, vg = [], []
            #delaunay = Delaunay(np.asarray(nodes))
            #for pt in grid:
                #μ, σ = interpolate_mean_and_sigma(pt, nodes, mu, P, delaunay, alpha)
                #μg.append(μ if μ is not None else 0.0)
                #vg.append((σ**2) if σ is not None else 0.0)
            #μg, vg = np.array(μg), np.array(vg)

            #plot_final_surface(grid, μg, prices, wages,
                               #filename=f"profit_surface_iter{iteration}.png")
            #plot_variance_surface(grid, vg, prices, wages,
                                  #filename=f"variance_surface_iter{iteration}.png")
            #import pandas as pd
            #df = pd.DataFrame(nodes, columns=["price", "wage"])
            #plot_voronoi_new(df, filename=f"voronoi_iter{iteration}.png")
            #plot_acquisition_trace(acq_trace, filename="acquisition_trace.png")

    print("=== Optimization complete ===")
    #plot_final_surface(grid, μg, prices, wages, filename="final_profit_surface.png")
    #plot_variance_surface(grid, vg, prices, wages, filename="final_variance_surface.png")
    plot_acquisition_trace(acq_trace, filename="final_acquisition_trace.png")
    #print("Saved all final plots.")
