import numpy as np
from scipy.spatial import Delaunay
from run_simulation import run_simulation

# New plotting imports
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
    bounds = [(1.0, 6.0), (0.0, 10.0)]
    cont = lhs_samples_2d(n_init, bounds)

    nodes, profits = [], []
    best_obs = -np.inf
    best_params = (None, None)

    for b_cont, p_cont in cont:
        b_int = int(np.clip(round(b_cont), 1, 6))
        y = run_simulation(p_cont, b_int)
        nodes.append([float(b_int), p_cont])
        profits.append(y)
        if y > best_obs:
            best_obs, best_params = y, (b_int, p_cont)

    mu = np.array(profits)
    σ0 = 1e4
    P = σ0 * np.eye(n_init)
    return nodes, mu, P, best_obs, best_params, R_noise

# ----------------------------------------------
# 2. Interpolation helpers & 3. Kalman update
# ----------------------------------------------
def barycentric_coordinates(tri_pts, x):
    T = np.vstack([tri_pts.T, np.ones(3)])
    v = np.append(x, 1.0)
    return np.linalg.solve(T, v)

def interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha=1.0):
    idx = int(delaunay.find_simplex(x.reshape(1, -1))[0])
    if idx < 0:
        return None, None
    verts = delaunay.simplices[idx]
    pts = np.asarray(nodes)[verts]
    λ = barycentric_coordinates(pts, x)
    mu_interp = sum(λ[i] * mu[verts[i]] for i in range(3))
    prior_var = sum((λ[i]**2) * P[verts[i], verts[i]] for i in range(3))
    dists = [np.linalg.norm(x - pts[i]) for i in range(3)]
    sigma = np.sqrt(prior_var + alpha * (max(dists)**2))
    return mu_interp, sigma

def one_iteration(nodes, mu, P, next_point, best_obs, best_params, R_noise, alpha=2.0):
    x_new = np.asarray(next_point)
    for i, pt in enumerate(nodes):
        if np.allclose(pt, x_new, atol=1e-6):
            # rank-one update
            y_new = run_simulation(x_new[1], int(x_new[0]))
            H = np.zeros((1, len(nodes))); H[0, i] = 1.0
            S = H @ P @ H.T + R_noise
            K = (P @ H.T) / S
            mu_post = mu + (K.flatten() * (y_new - H @ mu))
            P_post = P - (K @ H @ P)
            if y_new > best_obs:
                best_obs, best_params = y_new, (int(x_new[0]), float(x_new[1]))
            return nodes, mu_post, P_post, best_obs, best_params

    # new point
    if len(nodes) < 3:
        return nodes, mu, P, best_obs, best_params

    delaunay = Delaunay(np.asarray(nodes))
    mu0, sigma0 = interpolate_mean_and_sigma(x_new, nodes, mu, P, delaunay, alpha)
    if mu0 is None:
        return nodes, mu, P, best_obs, best_params

    N = len(nodes)
    # covariances
    tri_idx = int(delaunay.find_simplex(x_new.reshape(1, -1))[0])
    verts = delaunay.simplices[tri_idx]
    λ = barycentric_coordinates(np.asarray(nodes)[verts], x_new)
    cov_old = np.zeros(N)
    for jj, ii in enumerate(verts):
        cov_old[ii] = λ[jj] * P[ii, ii]

    mu_aug = np.concatenate([mu, [mu0]])
    P_aug = np.zeros((N+1, N+1))
    P_aug[:N, :N] = P
    P_aug[:N, N] = cov_old
    P_aug[N, :N] = cov_old
    P_aug[N, N] = sigma0**2

    y_new = run_simulation(x_new[1], int(x_new[0]))
    H = np.zeros((1, N+1)); H[0, N] = 1.0
    S = H @ P_aug @ H.T + R_noise
    K = (P_aug @ H.T) / S
    mu_post = mu_aug + (K.flatten() * (y_new - H @ mu_aug))
    P_post = P_aug - (K @ H @ P_aug)

    nodes_new = nodes + [[float(x_new[0]), float(x_new[1])]]
    if y_new > best_obs:
        best_obs, best_params = y_new, (int(x_new[0]), float(x_new[1]))
    return nodes_new, mu_post, P_post, best_obs, best_params

# ----------------------------------------------
# 4. UCB proposal (returns candidate + its UCB)
# ----------------------------------------------
def propose_next_ucb_discrete_b(nodes, mu, P, delaunay,
                                M_per_b=4000, kappa=5.0, alpha=3.0):
    """
    Evaluate UCB at each existing node AND at M_per_b random prices
    for each barista 1..6. Return the (b,p) with highest UCB and that UCB.
    """
    best_ucb = -np.inf
    best_cand = None

    # 1) First, consider all existing nodes:
    for (b, p), μ_val in zip(nodes, mu):
        # compute sigma at that exact node
        σ_val = np.sqrt(P[int(nodes.index([b, p])), int(nodes.index([b, p]))])
        ucb = μ_val + kappa * σ_val
        if ucb > best_ucb:
            best_ucb, best_cand = ucb, [float(b), float(p)]

    # 2) Then sample randomly as before:
    for b in range(1, 7):
        prices = np.random.uniform(0, 10, M_per_b)
        for p in prices:
            x = np.array([float(b), p])
            μ_val, σ_val = interpolate_mean_and_sigma(x, nodes, mu, P, delaunay, alpha)
            if μ_val is None:
                continue
            ucb = μ_val + kappa * σ_val
            if ucb > best_ucb:
                best_ucb, best_cand = ucb, [float(b), p]

    return best_cand, best_ucb

# ----------------------------------------------
# 5. Main optimization loop with plotting
# ----------------------------------------------
if __name__ == "__main__":
    # init
    n_initial, R_noise_est = 10, 450**2
    nodes, mu, P, best_obs, best_params, R_noise = initialize_with_lhs(
        n_init=n_initial, R_noise=R_noise_est
    )
    print("Initial best:", best_obs, "at", best_params)

    acq_trace = []
    n_iterations, kappa, alpha = 200, 5.0, 3.0

    for iteration in range(1, n_iterations + 1):
        delaunay = Delaunay(np.asarray(nodes))

        if np.random.rand() < 0.1:
            next_pt = [float(np.random.randint(1,7)), np.random.uniform(0,10)]
            ucb_val = None
        else:
            next_pt, ucb_val = propose_next_ucb_discrete_b(
                nodes, mu, P, delaunay, M_per_b=400, kappa=kappa, alpha=alpha
            )
        if ucb_val is not None:
            acq_trace.append(ucb_val)

        nodes, mu, P, best_obs, best_params = one_iteration(
            nodes, mu, P, next_pt, best_obs, best_params, R_noise, alpha=alpha
        )

        # logging
        if iteration % 10 == 0:
            print(f"Iter {iteration}: best observed = {best_obs:.3f} at {best_params}")

        # plotting every 50 iters and at end
        if iteration in [50, 100, 200]:
            print(f"--- Generating plots at iter {iteration} ---")
            baristas = np.arange(1,7)
            prices = np.linspace(0,10,100)
            grid = np.array([[b,p] for b in baristas for p in prices])

            # compute grid values
            μg, vg = [], []
            delaunay = Delaunay(np.asarray(nodes))
            for pt in grid:
                μ, σ = interpolate_mean_and_sigma(pt, nodes, mu, P, delaunay, alpha)
                μg.append(μ if μ is not None else 0.0)
                vg.append((σ**2) if σ is not None else 0.0)
            μg, vg = np.array(μg), np.array(vg)

            plot_final_surface(grid, μg, baristas, prices, 
                               filename=f"profit_surface_iter{iteration}.png")
            plot_variance_surface(grid, vg, baristas, prices,
                                  filename=f"variance_surface_iter{iteration}.png")
            import pandas as pd
            df = pd.DataFrame(nodes, columns=["baristas","price"])
            plot_voronoi_new(df, filename=f"voronoi_iter{iteration}.png")
            plot_acquisition_trace(acq_trace, filename="acquisition_trace.png")

    # final summary
    print("=== Optimization complete ===")
    # reuse same grid + μg/vg from iter 200
    plot_final_surface(grid, μg, baristas, prices, filename="final_profit_surface.png")
    plot_variance_surface(grid, vg, baristas, prices, filename="final_variance_surface.png")
    plot_acquisition_trace(acq_trace, filename="final_acquisition_trace.png")
    print("Saved all final plots.") 
