import numpy as np
import pandas as pd
from run_simulation import run_simulation
from plotting import plot_voronoi

baristas_range = np.arange(1, 6)
price_range = np.linspace(1, 10.0, 100)
grid = np.array([[b, p] for b in baristas_range for p in price_range])
n_iter = 2000
epsilon = 0.1

data = []
score_trace = []

# Initial random sample
idx = np.random.choice(len(grid))
b, p = grid[idx]
profit = run_simulation(p, b)
data.append((idx, profit))

for i in range(1, n_iter):
    if np.random.rand() < epsilon:
        # Explore: random
        idx = np.random.choice(len(grid))
    else:
        # Exploit: pick best so far
        idx = data[np.argmax([p for _, p in data])][0]

    b, p = grid[idx]
    profit = run_simulation(p, b)
    data.append((idx, profit))
    score_trace.append(profit)

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}] b={b}, p={p:.2f} -> profit={profit:.2f}")

# Save
df = pd.DataFrame([(grid[idx][0], grid[idx][1], prof) for idx, prof in data],
                  columns=["baristas", "price", "profit"])
df.to_csv("results/RawResults/results_eps_greedy.csv", index=False)
plot_voronoi(df, filename="results/voronoiDiagram/voronoi_eps_greedy.png")
print("✅ Epsilon-Greedy complete.")
