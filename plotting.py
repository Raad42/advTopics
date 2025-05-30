import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np

def plot_final_surface(df, baristas_range, price_range):
    from optimiser import interpolate_profit
    points = df[['baristas', 'price']].values
    profits = df['profit'].values
    grid_b, grid_p = np.meshgrid(baristas_range, price_range)
    grid_flat = np.c_[grid_b.ravel(), grid_p.ravel()]
    try:
        z_interp = interpolate_profit(points, profits, grid_flat)
    except:
        print("Delaunay interpolation failed.")
        return
    Z = z_interp.reshape(grid_p.shape)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_b, grid_p, Z, cmap='viridis', alpha=0.8)
    ax.scatter(df['baristas'], df['price'], df['profit'], color='r', label='Sampled points')
    ax.set_xlabel("Baristas")
    ax.set_ylabel("Price ($)")
    ax.set_zlabel("Profit")
    ax.set_title("Interpolated Profit Surface")
    plt.legend()
    plt.tight_layout()
    plt.savefig("profit_surface.png")
    print("Plot saved as 'profit_surface.png'")

def plot_variance_surface(grid, sigma_pred, baristas_range, price_range):
    grid_b, grid_p = np.meshgrid(baristas_range, price_range)
    Z = sigma_pred.reshape(grid_p.shape)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(grid_b, grid_p, Z, cmap='plasma', alpha=0.7)
    ax.set_title("Interpolated Uncertainty Surface (σ²)")
    plt.tight_layout()
    plt.savefig("variance_surface.png")
    print("Saved uncertainty surface as 'variance_surface.png'")

def plot_voronoi(df):
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
        ax.set_xlim([min(points[:, 0]) - 1, max(points[:, 0]) + 1])
        ax.set_ylim([min(points[:, 1]) - 1, max(points[:, 1]) + 1])
        ax.legend()
        plt.tight_layout()
        plt.savefig("voronoi_diagram.png")
        print("Saved Voronoi diagram as 'voronoi_diagram.png'")
    except Exception as e:
        print(f"Voronoi diagram failed: {e}")

def plot_acquisition_trace(acq_trace):
    plt.figure(figsize=(10, 4))
    plt.plot(acq_trace, color='darkblue')
    plt.xlabel("Iteration")
    plt.ylabel("Max Acquisition Score")
    plt.title("Acquisition Function Value Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("acquisition_trace.png")
    print("Saved acquisition trace as 'acquisition_trace.png'")
