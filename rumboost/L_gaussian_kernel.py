import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rumboost.utility_smoothing import data_leaf_value

def gaussian_kernel_smoother(x, y, bandwidth=1.0,
                               use_first_endpoint_weighting=True, first_endpoint_weight=0.05,
                               use_last_endpoint_weighting=True, last_endpoint_weight=0.05):
    """
    Apply Gaussian Kernel smoothing to raw data (x, y) with optional, independent endpoint replication.
    
    Parameters
    ----------
    x : array-like
        The x-values (raw data points).
    y : array-like
        The corresponding utility values.
    bandwidth : float, optional
        The kernel bandwidth controlling the smoothing (default: 1.0).
    use_first_endpoint_weighting : bool, optional
        If True, replicate the first data point to increase its weight.
    first_endpoint_weight : float, optional
        Fraction of total points to replicate for the first endpoint (default: 0.05).
    use_last_endpoint_weighting : bool, optional
        If True, replicate the last data point to increase its weight.
    last_endpoint_weight : float, optional
        Fraction of total points to replicate for the last endpoint (default: 0.05).
    
    Returns
    -------
    function
        A callable interpolator f(x_new) that returns the smoothed y-values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort data in ascending order
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Determine replication counts for first and last points separately
    N = len(x_sorted)
    rep_first = int(first_endpoint_weight * N) if use_first_endpoint_weighting else 0
    rep_last = int(last_endpoint_weight * N) if use_last_endpoint_weighting else 0
    if use_first_endpoint_weighting:
        rep_first = max(rep_first, 1)
    if use_last_endpoint_weighting:
        rep_last = max(rep_last, 1)
    
    # Replicate endpoints accordingly
    x_aug = np.concatenate([np.repeat(x_sorted[0], rep_first),
                            x_sorted,
                            np.repeat(x_sorted[-1], rep_last)])
    y_aug = np.concatenate([np.repeat(y_sorted[0], rep_first),
                            y_sorted,
                            np.repeat(y_sorted[-1], rep_last)])
    
    # Define the Gaussian kernel function
    def gaussian_kernel(z):
        return np.exp(-0.5 * z**2)
    
    # Smoothing: compute weighted average for each new x value using all augmented points
    def smoother(x_new):
        x_new = np.asarray(x_new)
        smoothed_vals = []
        for x0 in x_new:
            weights_kernel = gaussian_kernel((x0 - x_aug) / bandwidth)
            smoothed_value = np.sum(weights_kernel * y_aug) / np.sum(weights_kernel)
            smoothed_vals.append(smoothed_value)
        return np.array(smoothed_vals)
    
    # Compute smoothed values on a dense grid for interpolation
    x_dense = np.linspace(np.min(x_aug), np.max(x_aug), 500)
    y_dense = smoother(x_dense)
    
    def interpolator(x_new):
        x_new = np.asarray(x_new)
        return np.interp(x_new, x_dense, y_dense, left=y_dense[0], right=y_dense[-1])
    
    return interpolator

def plot_gaussian_smoothing(util_collection, utility_names, data_train, weights, bandwidth=1.0):
    """
    Plot Gaussian kernel smoothed utility functions along with the raw data points.
    This function mimics the LOESS plotting style (colors, fonts, markers, grid) from PCUF,
    but each (utility, feature) pair is plotted in a separate figure.
    
    Parameters
    ----------
    util_collection : dict
        Dictionary containing Gaussian kernel smoothing interpolators.
        Expected structure: {utility: {feature: interpolator_function}}.
    utility_names : dict
        Mapping of utility keys (e.g., '0', '1', etc.) to descriptive names.
    data_train : pandas.DataFrame
        The training dataset (used for extracting raw data points).
    weights : dict
        Dictionary containing weight data for each feature (used by data_leaf_value).
    bandwidth : float, optional
        The kernel bandwidth used (displayed in the legend, default: 1.0).
    """
    # Styling parameters (matching LOESS style)
    tex_fonts = {
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.labelpad": 1,
        "font.size": 7,
        "legend.fontsize": 6,
        "legend.fancybox": False,
        "legend.edgecolor": "inherit",
        "legend.borderaxespad": 0.4,
        "legend.borderpad": 0.4,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "xtick.major.pad": 0.5,
        "ytick.major.pad": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "scatter.edgecolors": "none",
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    sns.set_style("whitegrid")

    # Color mapping matching PCUF style
    color_map = {
        "walking": "b",
        "cycling": "r",
        "rail": "g",
        "driving": "orange",
        "swissmetro": "#6b8ba4"
    }

    # For each utility and feature, create a separate figure.
    for u in sorted(util_collection.keys()):
        for feature in util_collection[u]:
            plt.figure(figsize=(3.49, 2.09), dpi=1000)
            plt.title(f"{utility_names.get(u, u)} - {feature}", fontsize=7)
            
            # Retrieve raw data using data_leaf_value
            try:
                x_data, y_data = data_leaf_value(data_train[feature], weights[u][feature], "data_weighted")
            except Exception as e:
                print(f"Error retrieving data for utility {u}, feature {feature}: {e}")
                continue
            
            # Normalize raw data (subtract the first y value)
            y_data_norm = [val - y_data[0] for val in y_data]
            
            # Plot raw data points (black dots)
            plt.scatter(x_data, y_data_norm, color="k", s=4, alpha=1, edgecolors="none", label="Data")
            
            # Generate a dense grid over the x-data range and compute smoothed values
            x_dense = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_smooth = util_collection[u][feature](x_dense)
            y_smooth_norm = [val - y_smooth[0] for val in y_smooth]
            
            # Plot the smoothed curve with PCUF-matched style
            plt.plot(x_dense, y_smooth_norm, color=color_map.get(u, "#5badc7"), 
                     linewidth=0.8, label=f"Gaussian Kernel")
            
            # Set axis labels
            if "distance" in feature.lower():
                plt.xlabel(f"{feature} [km]", fontsize=7)
            else:
                plt.xlabel(feature, fontsize=7)
            plt.ylabel(f"{utility_names.get(u, u)} utility", fontsize=7)
            
            plt.legend(fontsize=6)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
