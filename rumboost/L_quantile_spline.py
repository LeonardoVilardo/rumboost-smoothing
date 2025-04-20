import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rumboost.utility_smoothing import data_leaf_value

def quantile_knots(x, num_knots=10, min_knots=3):
    """
    Compute quantile-based knots for a spline, ensuring a minimum number of knots.
    
    Parameters:
    -----------
    x : array-like
        Feature values (e.g., travel time, cost).
    num_knots : int, optional
        Target number of knots (default: 10).
    min_knots : int, optional
        Minimum number of knots to guarantee (default: 3).

    Returns:
    --------
    knots : np.array
        Array of selected knot positions.
    """
    unique_x = np.unique(x)
    
    if len(unique_x) < min_knots:
        return np.linspace(unique_x.min(), unique_x.max(), min_knots)
    
    return np.quantile(unique_x, np.linspace(0, 1, min(num_knots, len(unique_x))))


def cubic_spline_fixed_knots(x, y, num_knots=10):
    """
    Fit a cubic spline using quantile-based knots, ensuring the first and last data points are included exactly.

    Parameters:
    -----------
    x : array-like
        Feature values.
    y : array-like
        Corresponding utility values.
    num_knots : int, optional
        Number of knots (default: 10).

    Returns:
    --------
    spline : CubicSpline
        The fitted spline function.
    knots : np.array
        Knot positions.
    y_knots : np.array
        Corresponding y-values at knots.
    """
    x = np.array(x)
    y = np.array(y)

    # Ensure valid lengths
    min_length = min(len(x), len(y))
    x, y = x[:min_length], y[:min_length]

    # Sort data for interpolation
    sorted_indices = np.argsort(x)
    x_sorted, y_sorted = x[sorted_indices], y[sorted_indices]

    # Compute quantile-based knots but exclude the first and last positions
    quantile_knots_values = quantile_knots(x_sorted, num_knots - 2)  

    # Ensure first and last knots are exactly at the first and last data points
    first_knot, last_knot = x_sorted[0], x_sorted[-1]
    first_y, last_y = y_sorted[0], y_sorted[-1]

    # Combine knots
    all_knots = np.concatenate(([first_knot], quantile_knots_values, [last_knot]))
    
    # Ensure strict increasing sequence (remove duplicates or too-close values)
    min_gap = 1e-6  # Small threshold to avoid numerical issues
    filtered_knots = [all_knots[0]]  # Always include the first knot
    for k in all_knots[1:]:
        if k > filtered_knots[-1] + min_gap:  # Ensure it's strictly increasing
            filtered_knots.append(k)
    knots = np.array(filtered_knots)  # Convert back to array

    # Align y-values with knots, fixing the first and last y-values
    y_knots = np.array([first_y] + list(np.interp(knots[1:-1], x_sorted, y_sorted)) + [last_y])

    # Fit cubic spline
    spline = si.CubicSpline(knots, y_knots, bc_type="not-a-knot")

    print("Knots:", knots)
    print("Y Knots:", y_knots)

    return spline, knots, y_knots




def plot_quantile_spline(util_collection, util_collection_knots, utility_names, weights, data_train):
    """
    Plots quantile-spline-smoothed utility functions with original data points.

    Parameters:
    -----------
    util_collection : dict
        Dictionary containing quantile-based spline functions.
    util_collection_knots : dict
        Dictionary containing knots for each spline.
    utility_names : dict
        Dictionary mapping utility index to names.
    weights : dict
        Dictionary containing original data points for visualization.
    data_train : pandas.DataFrame
        The full training dataset.
    """
    tex_fonts = {
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
    sns.set_theme(style="whitegrid", rc=tex_fonts)

    num_vars = sum(len(vars) for vars in util_collection.values())
    fig, axes = plt.subplots(num_vars, 1, figsize=(8, 5 * num_vars), sharex=False)
    if num_vars == 1:
        axes = [axes]

    ax_idx = 0
    for utility, smoothed_vars in util_collection.items():
        for var in smoothed_vars:
            ax = axes[ax_idx]
            ax.set_title(f"{utility_names.get(utility, utility)} - {var}")

            interpolator = smoothed_vars[var]
            knots, y_knots = util_collection_knots[utility][var]  # Retrieve stored knots

            if not callable(interpolator):
                print(f"Skipping {var}: Not a callable interpolator.")
                continue

            # Extract data points
            x_points, y_points = data_leaf_value(data_train[var], weights[utility][var], "data_weighted")
            x_min, x_max = min(x_points), max(x_points)

            # Compute spline values
            x_smooth = np.linspace(x_min, x_max, 100)
            y_smooth = interpolator(x_smooth)

            # Plot original data
            ax.scatter(x_points, y_points, color="black", s=1.5, marker="o", edgecolors="black", label="Data")

            # Plot spline-smoothed curve
            ax.plot(x_smooth, y_smooth, label=f"Spline Smoothed - {var}", color="b")

            # Plot knots as orange circles
            ax.scatter(knots, y_knots, color="orange", s=40, label="Knots", edgecolors="black")

            ax.set_xlabel("Feature Value")
            ax.set_ylabel("Utility")
            ax.legend()
            ax.grid(True)

            print(f"{var}: Spline from {x_min} to {x_max} (Data extends to {x_max})")
            ax_idx += 1

    plt.tight_layout()
    plt.show()
