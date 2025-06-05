import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

# Updated plotting functions for vector-Kalman output

def plot_final_surface(grid, mu, baristas_range, price_range, filename="profit_surface.png"):  
    """
    Plot final estimated profit surface given mean vector `mu` on full grid.
    """
    # Reshape mu into meshgrid shape: rows=price, cols=baristas
    B, P = np.meshgrid(baristas_range, price_range)
    Z = mu.reshape(P.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(B, P, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel("Baristas")
    ax.set_ylabel("Price ($)")
    ax.set_zlabel("Estimated Profit")
    ax.set_title("Estimated Profit Surface")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved profit surface as '{filename}'")


def plot_variance_surface(grid, variances, baristas_range, price_range):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    X = np.array([pt[0] for pt in grid]).reshape(len(baristas_range), len(price_range)).T
    Y = np.array([pt[1] for pt in grid]).reshape(len(baristas_range), len(price_range)).T
    Z = variances.reshape(len(baristas_range), len(price_range)).T  # <-- key reshape fix

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.9)
    ax.set_title("Estimated Uncertainty Surface (Variance)")
    ax.set_xlabel("Baristas")
    ax.set_ylabel("Price ($)")
    ax.set_zlabel("Variance")
    plt.tight_layout()
    plt.savefig('plot.png')




def plot_voronoi(df, filename="voronoi_diagram.png"):  
    """
    Plot Voronoi diagram of sampled points (df with baristas, price).
    """
    points = df[['baristas', 'price']].values
    try:
        vor = Voronoi(points)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2)
        ax.plot(points[:, 0], points[:, 1], 'ko', label='Sampled Points')
        ax.set_xlabel('Baristas')
        ax.set_ylabel('Price')
        ax.set_title('Voronoi Diagram of Sampled Points')
        ax.set_xlim([points[:, 0].min()-1, points[:, 0].max()+1])
        ax.set_ylim([points[:, 1].min()-1, points[:, 1].max()+1])
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Saved Voronoi diagram as '{filename}'")
    except Exception as e:
        print(f"Voronoi diagram failed: {e}")


def plot_acquisition_trace(acq_trace, filename="acquisition_trace.png"):  
    """
    Plot the acquisition function's max score over iterations.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(acq_trace)
    plt.xlabel("Iteration")
    plt.ylabel("Max Acquisition Score")
    plt.title("Acquisition Function Value Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved acquisition trace as '{filename}'")
