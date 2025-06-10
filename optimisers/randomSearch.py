import numpy as np
import pandas as pd
from run_simulation import run_simulation
from plotting import plot_voronoi

baristas_range = np.arange(1, 6)
price_range = np.linspace(1, 10.0, 100)
grid = np.array([[b, p] for b in baristas_range for p in price_range])
n_samples = 2000

data = []
for i in range(n_samples):
    idx = np.random.choice(len(grid))
    b, p = grid[idx]
    profit = run_simulation(p, b)
    data.append((b, p, profit))

    if (i + 1) % 100 == 0:
        print(f"[{i + 1}] b={b}, p={p:.2f} -> profit={profit:.2f}")

df = pd.DataFrame(data, columns=["baristas", "price", "profit"])
df.to_csv("results/RawResults/results_random.csv", index=False)
plot_voronoi(df, filename="voronoi_random.png")
print("✅ Random search complete.")
