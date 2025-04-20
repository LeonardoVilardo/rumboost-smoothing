import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rumboost.utility_smoothing import data_leaf_value
from sklearn.isotonic import IsotonicRegression


def gaussian_kernel_smoother(x, y, bandwidth=1.0,
                           use_first_endpoint_weighting=True, first_endpoint_weight=0.05,
                           use_last_endpoint_weighting=True, last_endpoint_weight=0.05,
                           enforce_monotonic_decrease=True):
    """
    Apply Gaussian Kernel smoothing to raw data (x, y) with improved endpoint handling
    and optional monotonicity enforcement.
    
    Parameters
    ----------
    x : array-like
        The x-values (raw data points).
    y : array-like
        The corresponding utility values.
    bandwidth : float, optional
        The kernel bandwidth controlling the smoothing (default: 1.0).
    use_first_endpoint_weighting : bool, optional
        If True, apply boundary correction at the first data point.
    first_endpoint_weight : float, optional
        Weight for boundary correction at the first endpoint (default: 0.05).
    use_last_endpoint_weighting : bool, optional
        If True, apply boundary correction at the last data point.
    last_endpoint_weight : float, optional
        Weight for boundary correction at the last endpoint (default: 0.05).
    enforce_monotonic_decrease : bool, optional
        If True, enforce monotonically decreasing relationship using isotonic regression.
    
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
    
    # Define the Gaussian kernel function (simplified version suitable for our purposes)
    def gaussian_kernel(z):
        return np.exp(-0.5 * z**2)
    
    # Improved boundary handling function
    def boundary_correction_weights(t):
        """Compute boundary correction weights based on kernel support at each point."""
        weights = np.ones_like(t)
        # Calculate kernel sum for each point in t
        for i, x_val in enumerate(t):
            kernel_sum = gaussian_kernel((x_val - x_sorted) / bandwidth).sum()
            weights[i] = 1.0 / max(kernel_sum, 1e-8)  # Avoid division by zero
        return weights
    
    # Smoothing: compute weighted average for each new x value
    def smoother(x_new):
        x_new = np.asarray(x_new).reshape(-1)
        n_new = len(x_new)
        n_data = len(x_sorted)
        
        # Compute distances between new points and data points
        dist_matrix = np.zeros((n_new, n_data))
        for i in range(n_new):
            dist_matrix[i] = (x_new[i] - x_sorted) / bandwidth
        
        # Compute kernel values
        kernel_values = gaussian_kernel(dist_matrix)
        
        # Apply boundary corrections if requested
        if use_first_endpoint_weighting or use_last_endpoint_weighting:
            boundary_weights = boundary_correction_weights(x_new)
            
            # Apply endpoint-specific adjustments
            if use_first_endpoint_weighting:
                # Increase weight at first endpoint based on first_endpoint_weight
                start_mask = x_new <= x_sorted[0] + bandwidth
                boundary_weights[start_mask] *= (1 + first_endpoint_weight)
            
            if use_last_endpoint_weighting:
                # Increase weight at last endpoint based on last_endpoint_weight  
                end_mask = x_new >= x_sorted[-1] - bandwidth
                boundary_weights[end_mask] *= (1 + last_endpoint_weight)
                
            # Apply boundary weights to kernel values
            kernel_values = kernel_values * boundary_weights[:, np.newaxis]
        
        # Compute weighted averages
        weights_sum = kernel_values.sum(axis=1)
        weights_sum[weights_sum == 0] = 1  # Avoid division by zero
        smoothed_vals = (kernel_values @ y_sorted) / weights_sum
        
        return smoothed_vals
    
    # Compute smoothed values on a dense grid for interpolation
    x_dense = np.linspace(np.min(x_sorted), np.max(x_sorted), 500)
    y_dense = smoother(x_dense)
    
    # Apply isotonic regression if requested to enforce monotonicity
    if enforce_monotonic_decrease:
        iso_reg = IsotonicRegression(increasing=False)
        y_dense = iso_reg.fit_transform(x_dense, y_dense)
    
    def interpolator(x_new):
        x_new = np.asarray(x_new)
        return np.interp(x_new, x_dense, y_dense, left=y_dense[0], right=y_dense[-1])
    
    return interpolator


def plot_gaussian_smoothing(util_collection, utility_names, data_train, weights, bandwidth=1.0):
    """
    Plot Gaussian kernel smoothed utility functions along with the raw data points.
    This function mimics the LOESS plotting style (colors, fonts, markers, grid) from PCUF.
    
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

    # For each utility and feature, create a separate figure
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