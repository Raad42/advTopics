import matplotlib.pyplot as plt
import pandas as pd

def plot_benchmark(paths, labels):
    plt.figure(figsize=(10, 6))

    for path, label in zip(paths, labels):
        df = pd.read_csv(path)
        profits = df["profit"].values
        cum_best = [profits[:i+1].max() for i in range(len(profits))]
        plt.plot(cum_best, label=label)

    plt.title("Cumulative Best Profit vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Profit Found So Far")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png")
    

# Example usage
plot_benchmark(
    ["results_grid.csv", "results_random.csv", "results_eps_greedy.csv", "results.csv"],
    ["Grid Search", "Random Search", "Epsilon-Greedy", "Kalman Filter"]
)
