__version__ = "L_pcuf_quantile v1.0"

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
from rumboost.utility_smoothing import data_leaf_value

# --- Helper functions for adaptive quantile knots ---

def determine_num_knots(data, min_knots=4, max_knots=10):
    """
    Determine an adaptive number of knots based on data spread.
    Combines IQR and standard deviation relative to the data range.
    """
    data = np.asarray(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    std_dev = np.std(data)
    range_val = np.max(data) - np.min(data)
    spread_score = (iqr + std_dev) / max(1e-6, range_val)
    num_knots = int(min_knots + (max_knots - min_knots) * spread_score)
    return int(np.clip(num_knots, min_knots, max_knots))

def quantile_knots(data, num_knots):
    """
    Compute quantile‐based knots from the unique values of data.
    """
    data = np.sort(np.unique(data))
    return np.quantile(data, np.linspace(0, 1, num_knots))

# --- PCUF Smoothers ---

def pcuf_histogram_smoother(x, y, *, x_knots=None, y_knots=None, 
                            knot_direction="horizontal", 
                            min_knots=6, max_knots=22, debug=False,
                            extra_knot_factor=3):
    """
    PCUF histogram-based smoother.
    
    If x_knots and y_knots are not provided, this function computes an adaptive set
    of knots based on the chosen knot_direction: if "horizontal", knots are computed
    from x and then y_knots are obtained via interpolation from (x,y); if "vertical",
    knots are computed from y and then corresponding x positions are derived.
    
    Then, additional (extra) knots are inserted in intervals where the absolute slope 
    between consecutive knots is greater than extra_knot_factor times the median slope.
    
    Finally, a monotonic spline is fitted through these (knot, y) pairs using 
    PchipInterpolator.
    
    Parameters
    ----------
    x : array-like
        Feature values.
    y : array-like
        Corresponding utility values.
    x_knots : array-like, optional
        If provided, these are used as the x positions for the knots.
    y_knots : array-like, optional
        If provided, these are used as the utility values at the knots.
    knot_direction : {"horizontal", "vertical"}, optional
        If "horizontal" (default), compute quantiles on x; if "vertical", compute on y.
    min_knots : int, optional
        Minimum number of knots.
    max_knots : int, optional
        Maximum number of knots.
    debug : bool, optional
        If True, prints debugging output.
    extra_knot_factor : float, optional
        An extra knot is inserted in an interval if the local slope is greater than 
        extra_knot_factor * median_slope.
    
    Returns
    -------
    smoother : callable
        Function f(x_new) that evaluates the fitted spline.
    knots : np.ndarray
        The x positions used for the knots.
    y_knots : np.ndarray
        The corresponding utility values at those knots.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Compute adaptive knots if not provided.
    if x_knots is None or y_knots is None:
        if knot_direction == "horizontal":
            data_for_knots = x
        elif knot_direction == "vertical":
            data_for_knots = y
        else:
            raise ValueError("Invalid knot_direction; choose 'horizontal' or 'vertical'")
        
        num_knots = determine_num_knots(data_for_knots, min_knots, max_knots)
        base_knots = quantile_knots(data_for_knots, num_knots)
        
        if knot_direction == "horizontal":
            knots = np.copy(base_knots)
            knots[0] = np.min(x)
            knots[-1] = np.max(x)
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]
            y_knots = np.interp(knots, x_sorted, y_sorted)
        else:  # vertical
            knots_y = np.copy(base_knots)
            knots_y[0] = np.min(y)
            knots_y[-1] = np.max(y)
            sort_idx = np.argsort(y)
            y_sorted = y[sort_idx]
            x_sorted = x[sort_idx]
            knots = np.interp(knots_y, y_sorted, x_sorted)
            y_knots = knots_y
    else:
        knots = np.asarray(x_knots)
        y_knots = np.asarray(y_knots)
    
    # Insert extra knots in regions with steep slope.
    if extra_knot_factor is not None and len(knots) >= 2:
        dx = np.diff(knots)
        dy = np.diff(y_knots)
        slopes = np.abs(dy / dx)
        median_slope = np.median(slopes)
        new_knots = []
        new_y_knots = []
        for i in range(len(knots) - 1):
            new_knots.append(knots[i])
            new_y_knots.append(y_knots[i])
            if slopes[i] > extra_knot_factor * median_slope:
                # Insert an extra knot at the midpoint.
                extra_knot = (knots[i] + knots[i+1]) / 2.0
                extra_y = (y_knots[i] + y_knots[i+1]) / 2.0
                new_knots.append(extra_knot)
                new_y_knots.append(extra_y)
        new_knots.append(knots[-1])
        new_y_knots.append(y_knots[-1])
        # Sort and remove duplicates.
        new_knots = np.unique(new_knots)
        new_y_knots = np.interp(new_knots, knots, y_knots)
        knots = new_knots
        y_knots = new_y_knots
        if debug:
            print("[DEBUG] Extra knots inserted based on steep slope:")
            print(f"  New knots (x): {knots}")
            print(f"  New y_knots: {y_knots}")
    
    if debug:
        print("[DEBUG] pcuf_histogram_smoother:")
        print(f"  knot_direction: {knot_direction}")
        print(f"  Knots (x): {knots}")
        print(f"  y_knots: {y_knots}")
    
    # Fit a monotonic spline (PCHIP preserves monotonicity)
    spline = PchipInterpolator(knots, y_knots, extrapolate=True)
    
    def smoother(x_new):
        return spline(x_new)
    
    # Attach knots to the smoother for plotting purposes.
    smoother.knots = knots
    smoother.y_knots = y_knots
    
    return smoother, knots, y_knots

def pcuf_raw_smoother(x, y, *, x_knots=None, y_knots=None,
                      knot_direction="horizontal", 
                      min_knots=6, max_knots=22, debug=False):
    """
    PCUF raw-data smoother.
    
    Similar to pcuf_histogram_smoother but fits directly to the raw (x, y) pairs.
    It computes adaptive quantile-based knots on either x or y (depending on
    `knot_direction`) and interpolates raw data at those positions.
    
    Parameters
    ----------
    x : array-like
        Feature values (e.g. time, distance).
    y : array-like
        Corresponding utility values (black dots).
    x_knots : array-like, optional
        Predefined knot positions (on x-axis).
    y_knots : array-like, optional
        Predefined utility values at knots.
    knot_direction : {"horizontal", "vertical"}, optional
        If "horizontal" (default), compute quantiles on x.
        If "vertical", compute on y and interpolate for x.
    min_knots : int, optional
        Minimum number of knots to use.
    max_knots : int, optional
        Maximum number of knots to use.
    debug : bool, optional
        Print internal details if True.
    
    Returns
    -------
    smoother : callable
        Interpolator function f(x_new) for the smoothed utility.
    knots : np.ndarray
        Knot x positions.
    y_knots : np.ndarray
        Knot y values (utilities).
    """
    # For now, the raw-data smoother uses the same logic as the histogram smoother.
    return pcuf_histogram_smoother(x, y, x_knots=x_knots, y_knots=y_knots,
                                   knot_direction=knot_direction,
                                   min_knots=min_knots, max_knots=max_knots,
                                   debug=debug)

def plot_pcuf_smoothing(util_collection, utility_names, data_train, weights):
    """
    Plot the PCUF-smoothed utility functions alongside the raw data.
    Assumes util_collection is a dict {utility: {feature: smoother}}.
    
    Parameters
    ----------
    util_collection : dict
        Dictionary mapping utility indices to dictionaries mapping feature names to smoother functions.
    utility_names : dict
        Mapping of utility keys (e.g., '0', '1') to display names.
    data_train : pandas.DataFrame
        The training dataset containing raw feature values.
    weights : dict
        Dictionary with weight data (if needed for data extraction).
    """
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
    }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    sns.set_style("whitegrid")
    
    color_map = {"walking": "b", "cycling": "r", "rail": "g", 
                 "driving": "orange", "swissmetro": "#6b8ba4"}
    
    total_plots = sum(len(util_collection[u]) for u in util_collection)
    if total_plots == 0:
        raise ValueError("No utilities to plot; check your util_collection.")
    
    fig, axes = plt.subplots(total_plots, 1, figsize=(3.49, 2.09 * total_plots), dpi=1000)
    if total_plots == 1:
        axes = [axes]
     
    ax_idx = 0
    for u in sorted(util_collection.keys()):
        for feature in util_collection[u]:
            ax = axes[ax_idx]
            try:
                x_data, y_data = data_leaf_value(data_train[feature], weights[u][feature], "data_weighted")
            except Exception as e:
                print(f"[ERROR] Utility {u}, feature {feature}: {e}")
                continue
            
            # Normalize raw data so that the first value is zero.
            y_data_norm = y_data - y_data[0]
            ax.scatter(x_data, y_data_norm, color="k", s=4, alpha=1, 
                       edgecolors="none", label="Data")
            
            # Evaluate the smoother on a dense grid.
            x_dense = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_smooth = util_collection[u][feature](x_dense)
            # Normalize the smoothed output so that its first value is zero.
            y_smooth_norm = y_smooth - y_smooth[0]
            ax.plot(x_dense, y_smooth_norm, color=color_map.get(u, "#5badc7"),
                    linewidth=0.8, label="PCUF Smoothed")
            
            # Add knots if available.
            smoother_func = util_collection[u][feature]
            if hasattr(smoother_func, "knots") and hasattr(smoother_func, "y_knots"):
                x_knots = smoother_func.knots
                y_knots = smoother_func.y_knots
                # Normalize knot utilities with the same offset as the raw data.
                y_knots_norm = y_knots - y_data[0]
                ax.scatter(x_knots, y_knots_norm, color="#BE5C23", 
                           s=1, zorder=3, label="Knots")
            
            ax.set_xlabel(f"{feature} [km]" if "distance" in feature else feature)
            ax.set_ylabel(f"{utility_names.get(u, u)} utility")
            
            # Consolidate legend entries.
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(True)
            ax_idx += 1
    
    plt.tight_layout()
    plt.show()
