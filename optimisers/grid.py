import numpy as np
import pandas as pd
from run_simulation import run_simulation
from plotting import plot_voronoi

baristas_range = np.arange(1, 6)
price_range = np.linspace(1, 10.0, 100)
grid = np.array([[b, p] for b in baristas_range for p in price_range])

data = []
for idx, (b, p) in enumerate(grid):
    profit = run_simulation(p, b)
    data.append((b, p, profit))
    if (idx + 1) % 100 == 0:
        print(f"[{idx + 1}/{len(grid)}] b={b}, p={p:.2f} -> profit={profit:.2f}")

df = pd.DataFrame(data, columns=["baristas", "price", "profit"])
df.to_csv("results_grid.csv", index=False)
plot_voronoi(df, filename="voronoi_grid.png")
print("✅ Grid search complete.")
