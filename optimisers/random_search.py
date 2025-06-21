import time
import numpy as np
import pandas as pd
from run_simulation import run_simulation

def run_random_search(n_samples=200):
    price_range = np.linspace(1.0, 10.0, 20)
    wage_range  = np.linspace(15.0, 50.0, 20)
    grid = np.array([[p, w] for p in price_range for w in wage_range])

    data = []
    start_time = time.time()

    for i in range(n_samples):
        idx = np.random.choice(len(grid))
        price, wage = grid[idx]

        profits = [run_simulation(price, wage) for _ in range(3)]  # Run 3 times
        avg_profit = np.mean(profits)

        data.append((price, wage, avg_profit))

        if (i + 1) % 100 == 0:
            print(f"[Random] {i + 1}/{n_samples} price={price:.2f}, wage={wage:.2f} -> avg_profit={avg_profit:.2f}")

    end_time = time.time()
    elapsed = end_time - start_time

    df = pd.DataFrame(data, columns=["price", "wage", "profit"])
    df.to_csv("results/RawResults/results_random.csv", index=False)

    best_idx = df["profit"].idxmax()
    best_price = df.loc[best_idx, "price"]
    best_wage = df.loc[best_idx, "wage"]
    best_profit = df.loc[best_idx, "profit"]

    # Calculate 95% threshold
    threshold_95 = 0.95 * best_profit

    # Find first iteration reaching 95% of max profit
    iter_to_95 = None
    for i, val in enumerate(df["profit"]):
        if val >= threshold_95:
            iter_to_95 = i + 1  # 1-based iteration
            break

    # Approximate time to reach 95% linearly
    time_to_95 = None
    if iter_to_95 is not None:
        time_to_95 = elapsed * (iter_to_95 / n_samples)

    return {
        "best_price": best_price,
        "best_wage": best_wage,
        "mean_profit": best_profit,
        "total_time": elapsed,
        "iter_to_95": iter_to_95,
        "time_to_95": time_to_95,
    }

if __name__ == "__main__":
    result = run_random_search(n_samples=2000)
    print("Random Search Result:")
    print(result)
