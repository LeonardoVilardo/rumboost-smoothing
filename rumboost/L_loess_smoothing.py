"""
LOESS (Locally Estimated Scatterplot Smoothing) Implementation for RUMBoost

This module implements LOESS smoothing as described in:
- James et al. (2023). "An Introduction to Statistical Learning"

LOESS performs non-parametric smoothing by fitting a smoothed curve directly 
from the shape of the data using local linear regressions with a tricube kernel.
Monotonicity is enforced post-smoothing using isotonic regression.
"""

__version__ = "L_loess v_corrected_1.0"

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from rumboost.utility_smoothing import data_leaf_value

def loess_smoother(x, y, frac=0.2, 
                   use_endpoint_weighting=True, 
                   endpoint_weight=0.05, 
                   enforce_monotonicity=True,
                   debug=False):
    """
    Apply LOESS smoothing with robust iterations disabled (it=0) on all raw points.
    Optionally increases weight of endpoints and enforces monotonicity via isotonic regression.
    
    LOESS performs local linear regressions using a tricube kernel that assigns higher 
    weights to closer observations. The smoothing span is controlled by the frac parameter.
    
    Parameters
    ----------
    x : array-like
        The x-values (all raw data points).
    y : array-like
        The corresponding utility values.
    frac : float, optional
        Fraction of data points used in each local regression window.
        Smaller values capture fine detail but risk overfitting.
        Larger values yield smoother global trends.
    use_endpoint_weighting : bool, optional
        If True, replicate the first and last points to increase their weight.
    endpoint_weight : float, optional
        Fraction of total points to replicate for each endpoint.
    enforce_monotonicity : bool, optional
        If True, applies isotonic regression to enforce a decreasing function.
    debug : bool, optional
        If True, prints debugging information.
    
    Returns
    -------
    callable
        A function f(x_new) that returns LOESS-smoothed y-values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort data by x-values to ensure numerical stability
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Optional endpoint weighting by replication
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
    
    # Apply LOESS using statsmodels with robust iterations disabled
    loess_result = sm.nonparametric.lowess(
        y_augmented, x_augmented, 
        frac=frac, 
        it=0,  # No robust iterations as specified
        return_sorted=True
    )
    
    x_loess = loess_result[:, 0]
    y_loess = loess_result[:, 1]
    
    # Enforce monotonicity using isotonic regression if requested
    if enforce_monotonicity:
        ir = IsotonicRegression(increasing=False)
        y_loess = ir.fit_transform(x_loess, y_loess)
    
    if debug:
        print(f"[DEBUG] LOESS smoother:")
        print(f"  Original data: n={len(x)}, x_range=({np.min(x):.4f}, {np.max(x):.4f})")
        print(f"  Augmented data: n={len(x_augmented)}")
        print(f"  LOESS output: n={len(x_loess)}")
        print(f"  Smoothing span: frac={frac}")
        if use_endpoint_weighting:
            print(f"  Endpoint replication: {rep_count} points per endpoint")
        print(f"  Monotonicity enforced: {enforce_monotonicity}")
    
    def interpolator(x_new):
        """
        Convert the discrete LOESS output into a continuous function using linear interpolation.
        """
        x_new = np.asarray(x_new)
        return np.interp(x_new, x_loess, y_loess, left=y_loess[0], right=y_loess[-1])
    
    return interpolator


def plot_loess_smoothing(util_collection, utility_names, data_train, weights):
    """
    Plot the LOESS-smoothed utility functions alongside raw data.
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
            
            # Get the LOESS interpolator
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
            
            # Plot LOESS smoothed curve
            ax.plot(x_dense, y_smooth_norm,
                   color=color_map.get(utility, "#5badc7"),
                   linewidth=0.8, label="LOESS")
            
            # Set axis labels
            ax.set_xlabel(f"{feature} [km]" if "distance" in feature else feature)
            ax.set_ylabel(f"{utility_names.get(utility, utility)} utility")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()