import time
import numpy as np
import pandas as pd
from run_simulation import run_simulation

def run_grid_search():
    price_range = np.linspace(1.0, 10.0, 20)    # Price ∈ [1, 10]
    wage_range  = np.linspace(15.0, 50.0, 20)   # Wage  ∈ [15, 50]
    grid = np.array([[p, w] for p in price_range for w in wage_range])

    data = []
    start_time = time.time()

    for idx, (p, w) in enumerate(grid):
        profits = [run_simulation(p, w) for _ in range(3)]  # Run 3 times
        avg_profit = np.mean(profits)
        data.append((p, w, avg_profit))
        if (idx + 1) % 100 == 0:
            print(f"[Grid] {idx + 1}/{len(grid)} price={p:.2f}, wage={w:.2f} -> avg_profit={avg_profit:.2f}")

    end_time = time.time()
    elapsed = end_time - start_time

    df = pd.DataFrame(data, columns=["price", "wage", "profit"])
    df.to_csv("results/RawResults/results_grid.csv", index=False)

    best_idx = df["profit"].idxmax()
    best_price = df.loc[best_idx, "price"]
    best_wage = df.loc[best_idx, "wage"]
    best_profit = df.loc[best_idx, "profit"]

    # 95% threshold
    threshold_95 = 0.95 * best_profit

    # Find first point reaching 95% of max profit
    iter_to_95 = None
    for i, val in enumerate(df["profit"]):
        if val >= threshold_95:
            iter_to_95 = i + 1
            break

    # Estimate time to 95%
    time_to_95 = None
    if iter_to_95 is not None:
        time_to_95 = elapsed * (iter_to_95 / len(df))

    return {
        "best_price": best_price,
        "best_wage": best_wage,
        "mean_profit": best_profit,
        "total_time": elapsed,
        "iter_to_95": iter_to_95,
        "time_to_95": time_to_95,
    }

if __name__ == "__main__":
    result = run_grid_search()
    print("Grid Search Result:")
    print(result)
