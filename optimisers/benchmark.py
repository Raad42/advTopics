import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from optimisers.optimiser import run_bayes_opt
from optimisers.grid_search import run_grid_search
from optimisers.random_search import run_random_search

def run_multiple_times(func, bounds=None, n_runs=20):
    profits = []
    times = []
    conv_metrics = []  # (time_to_best, iter_to_best, time_to_95pct, iter_to_95pct)
    sim_times = []
    comp_times = []
    
    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs} of {func.__name__}")
        start = time.time()
        if bounds is not None:
            res = func(bounds=bounds)
        else:
            res = func()
        elapsed = time.time() - start
        profits.append(res['mean_profit'])
        times.append(elapsed)
        conv_metrics.append((
            res.get('time_to_best'),
            res.get('iter_to_best'),
            res.get('time_to_95pct'),
            res.get('iter_to_95pct')
        ))
        if 'sim_time_total' in res:
            sim_times.append(res['sim_time_total'])
        if 'comp_time_total' in res:
            comp_times.append(res['comp_time_total'])

    return (
        np.array(profits),
        np.array(times),
        conv_metrics,
        np.array(sim_times) if sim_times else None,
        np.array(comp_times) if comp_times else None 
    )


def plot_cdfs(data, labels, filename):
    plt.figure(figsize=(8, 6))
    for vals, lbl in zip(data, labels):
        s = np.sort(vals)
        cdf = np.arange(1, len(s) + 1) / len(s)
        plt.step(s, cdf, label=lbl)
    plt.legend()
    plt.xlabel('Best Profit')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    runs = 20
    bounds = np.array([[1.0, 10.0], [15.0, 50.0]])

    bo_p, bo_t, bo_conv, bo_sim, bo_comp = run_multiple_times(run_bayes_opt, bounds, runs)
    gr_p, gr_t, gr_conv, gr_sim, _       = run_multiple_times(run_grid_search, None, runs)
    rd_p, rd_t, rd_conv, rd_sim, _       = run_multiple_times(run_random_search, None, runs)

    plot_cdfs([bo_p, gr_p, rd_p], ['BO', 'Grid', 'Random'], 'results/cdf.png')

    print("\nMann-Whitney U Tests:")
    for a, b, name in [(bo_p, rd_p, 'BO vs Random'), (bo_p, gr_p, 'BO vs Grid')]:
        u, p = mannwhitneyu(a, b, alternative='two-sided')
        print(f"{name}: U={u:.2f}, p={p:.4f}")

    print("\nSummary Table (averages across runs):")
    def summarize(label, profits, times, convs, sim_times=None, comp_times=None):
        to_best = [t for t, _, _, _ in convs if t is not None]
        it_best = [i for _, i, _, _ in convs if i is not None]
        to_95   = [t for _, _, t, _ in convs if t is not None]
        it_95   = [i for _, _, _, i in convs if i is not None]
        avg_sim = sim_times.mean() if sim_times is not None else None
        avg_comp = comp_times.mean() if comp_times is not None else None

        print(f"\n{label} Optimizer:")
        print(f"  Avg Profit:            {profits.mean():.2f}")
        print(f"  Avg Time to 95%:       {np.mean(to_95):.2f}s over {len(to_95)} runs")
        print(f"  Avg Iter to 95%:       {np.mean(it_95):.1f} iters")
        print(f"  Avg Time to Best:      {np.mean(to_best):.2f}s over {len(to_best)} runs")
        print(f"  Avg Iter to Best:      {np.mean(it_best):.1f} iters")
        if avg_sim is not None:
            print(f"  Avg Simulation Time:   {avg_sim:.2f}s")
        if avg_comp is not None:
            print(f"  Avg Computation Time:  {avg_comp:.2f}s")

    summarize("Bayesian", bo_p, bo_t, bo_conv, bo_sim, bo_comp)
    summarize("Grid", gr_p, gr_t, gr_conv, gr_sim)
    summarize("Random", rd_p, rd_t, rd_conv, rd_sim)
