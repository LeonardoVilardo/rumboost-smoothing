"""
Distance-based Moving Average Implementation for RUMBoost

This module implements a distance-based moving average smoothing technique that
constructs smoothed utility curves by averaging gradient boosted utility values 
within a specified x-axis radius. Unlike index-based methods that average a fixed 
number of points, this approach averages points within a consistent distance window,
resulting in visually smoother curves.

The window radius adapts dynamically to local data density:
- Narrow radii in data-dense regions to capture detail  
- Broader radii in sparse regions to reduce noise

Window sizes are specified as scales (fractions of the domain range) rather than 
absolute values, allowing consistent behavior across variables with different units.

References:
- Oluleye et al. (2023). "Exploratory Data Analysis", EDA Techniques.
"""

__version__ = "L_moving_average v_corrected_1.0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from rumboost.utility_smoothing import data_leaf_value

def moving_average_smoother(x, y, window_radius_scale=0.05, 
                          use_adaptive_window=True,
                          min_radius_scale=0.03, 
                          max_radius_scale=0.2, 
                          scaling_factor=0.5,
                          use_endpoint_weighting=True,
                          endpoint_weight=0.05,
                          enforce_monotonicity=True,
                          debug=False):
    """
    Apply moving average smoothing with distance-based windows.
    Instead of averaging a fixed number of points, averages points within a 
    specified x-axis radius around each point. Window sizes are specified as
    fractions of the domain range rather than absolute values.
    
    Parameters
    ----------
    x : array-like
        The x-values (all raw GBUV points).
    y : array-like
        The corresponding utility values.
    window_radius_scale : float, optional
        Fixed radius as fraction of domain range.
        Only used if use_adaptive_window=False.
    use_adaptive_window : bool, optional
        If True, use density-based adaptive window sizing.
    min_radius_scale : float, optional
        Minimum window radius as fraction of domain range.
    max_radius_scale : float, optional
        Maximum window radius as fraction of domain range.
    scaling_factor : float, optional
        Global scaling factor (c) for adaptive window computation.
    use_endpoint_weighting : bool, optional
        If True, apply endpoint replication to reduce boundary instability.
    endpoint_weight : float, optional
        Fraction of total points to replicate for each endpoint.
    enforce_monotonicity : bool, optional
        If True, applies isotonic regression to enforce a decreasing function.
    debug : bool, optional
        If True, prints debugging information.
    
    Returns
    -------
    callable
        A function f(x_new) that returns moving average smoothed y-values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort data for numerical stability
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Calculate domain range for converting scales to absolute values
    x_range = np.max(x_sorted) - np.min(x_sorted)
    
    # Convert scale parameters to absolute values
    window_radius = x_range * window_radius_scale
    min_radius = x_range * min_radius_scale
    max_radius = x_range * max_radius_scale
    
    # Apply endpoint replication if requested
    if use_endpoint_weighting:
        n_points = len(x_sorted)
        rep_count = max(1, int(endpoint_weight * n_points))
        
        x_augmented = np.concatenate([
            np.repeat(x_sorted[0], rep_count),
            x_sorted,
            np.repeat(x_sorted[-1], rep_count)
        ])
        y_augmented = np.concatenate([
            np.repeat(y_sorted[0], rep_count),
            y_sorted,
            np.repeat(y_sorted[-1], rep_count)
        ])
    else:
        x_augmented = x_sorted
        y_augmented = y_sorted
    
    # Compute adaptive window radii based on data density
    if use_adaptive_window:
        # Calculate inter-point spacing
        dx = np.diff(x_augmented)
        # Add a last element to maintain same length
        dx = np.append(dx, dx[-1])
        
        # Compute density-based window radii: R_i = clip(c × Δx_i, R_min, R_max)
        adaptive_radii = np.clip(scaling_factor * dx, min_radius, max_radius)
    else:
        adaptive_radii = np.full(len(x_augmented), window_radius)
    
    # Apply distance-based moving average
    y_smooth = np.zeros_like(y_augmented, dtype=np.float64)
    
    for i in range(len(y_augmented)):
        radius = adaptive_radii[i]
        current_x = x_augmented[i]
        
        # Find all points within the x-radius
        mask = np.abs(x_augmented - current_x) <= radius
        
        # Compute average of points within window
        if np.any(mask):
            y_smooth[i] = np.mean(y_augmented[mask])
        else:
            # Fallback if no points in window (unlikely but possible)
            y_smooth[i] = y_augmented[i]
    
    # Remove endpoint replication if applied
    if use_endpoint_weighting:
        y_smooth = y_smooth[rep_count:-rep_count]
        x_final = x_sorted
    else:
        x_final = x_augmented
    
    # Enforce monotonicity using isotonic regression
    if enforce_monotonicity:
        ir = IsotonicRegression(increasing=False)
        y_smooth = ir.fit_transform(x_final, y_smooth)
    
    if debug:
        print(f"[DEBUG] Moving average smoother:")
        print(f"  Original data: n={len(x)}, x_range=({np.min(x):.4f}, {np.max(x):.4f})")
        print(f"  Domain range: {x_range:.4f}")
        print(f"  Window parameters:")
        print(f"    - window_radius_scale: {window_radius_scale} -> {window_radius:.4f}")
        print(f"    - min_radius_scale: {min_radius_scale} -> {min_radius:.4f}")
        print(f"    - max_radius_scale: {max_radius_scale} -> {max_radius:.4f}")
        if use_adaptive_window:
            print(f"  Adaptive radii: min={np.min(adaptive_radii):.3f}, max={np.max(adaptive_radii):.3f}, mean={np.mean(adaptive_radii):.3f}")
        if use_endpoint_weighting:
            print(f"  Endpoint replication: {rep_count} points per endpoint")
        print(f"  Monotonicity enforced: {enforce_monotonicity}")
    
    def interpolator(x_new):
        """
        Convert the smoothed output into a continuous function using linear interpolation.
        """
        x_new = np.asarray(x_new)
        return np.interp(x_new, x_final, y_smooth, left=y_smooth[0], right=y_smooth[-1])
    
    return interpolator


def plot_moving_average_smoothing(util_collection, utility_names, data_train, weights):
    """
    Plot the moving average-smoothed utility functions alongside raw data.
    Each plot is rendered in its own figure for individual export.
    
    Parameters
    ----------
    util_collection : dict
        {utility: {feature: smoother}}, each 'smoother' is a callable f(x).
    utility_names : dict
        Maps utility keys to display names.
    data_train : pd.DataFrame
        The training data.
    weights : dict
        The leaf-based utilities for each feature.
    """
    # Define styling parameters to match PCUF plots
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
    
    # Color mapping for different utilities
    color_map = {
        "walking": "b",
        "cycling": "r",
        "rail": "g",
        "driving": "orange",
        "swissmetro": "#6b8ba4"
    }
    
    for utility in sorted(util_collection.keys()):
        for feature in util_collection[utility]:
            plt.figure(figsize=(3.49, 2.09), dpi=1000)
            ax = plt.gca()
            
            # Get the interpolator
            interpolator = util_collection[utility][feature]
            if not callable(interpolator):
                print(f"Skipping {feature}: not a callable interpolator.")
                plt.close()
                continue
            
            # Extract raw data points for plotting
            try:
                x_data, y_data = data_leaf_value(
                    data_train[feature],
                    weights[utility][feature],
                    technique="data_weighted"
                )
            except Exception as e:
                print(f"[ERROR] Utility {utility}, feature {feature}: {e}")
                plt.close()
                continue
            
            # Normalize both raw data and smoothed curve to start at 0
            y_data_norm = y_data - y_data[0]
            
            # Plot raw data points
            ax.scatter(x_data, y_data_norm, color="k", s=4, alpha=1,
                      edgecolors="none", label="Data")
            
            # Generate dense grid for smooth curve
            x_dense = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_smooth = interpolator(x_dense)
            y_smooth_norm = y_smooth - y_smooth[0]
            
            # Plot smoothed curve
            ax.plot(x_dense, y_smooth_norm,
                   color=color_map.get(utility, "#5badc7"),
                   linewidth=0.8, label="Moving Avg")
            
            # Set axis labels
            ax.set_xlabel(f"{feature} [km]" if "distance" in feature else feature)
            ax.set_ylabel(f"{utility_names.get(utility, utility)} utility")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()