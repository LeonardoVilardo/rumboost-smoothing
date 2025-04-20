__version__ = "L_pcuf_adapted v1.0"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rumboost.utility_smoothing import monotone_spline, smooth_predict, data_leaf_value
from rumboost.metrics import cross_entropy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def determine_num_knots_adaptive(data, min_knots=6, max_knots=12):
    """
    Determine the number of knots based on data spread score as described in the paper.
    
    Parameters
    ----------
    data : array-like
        The feature values.
    min_knots : int, optional
        Minimum number of knots (default: 6).
    max_knots : int, optional
        Maximum number of knots (default: 12).
    
    Returns
    -------
    num_knots : int
        Adaptive number of knots.
    """
    data = np.asarray(data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    std_dev = np.std(data)
    range_val = np.max(data) - np.min(data)
    
    # Handle edge cases
    if range_val < 1e-6:
        return min_knots
    
    spread_score = (iqr + std_dev) / range_val
    num_knots = int(min_knots + (max_knots - min_knots) * spread_score)
    return np.clip(num_knots, min_knots, max_knots)


def quantile_based_knots(data, min_knots=6, max_knots=12):
    """
    Compute knot positions based on data quantiles using adaptive number of knots.
    Ensures that the first and last data points are always included as knots.
    
    Parameters
    ----------
    data : array-like
        The feature values.
    min_knots : int, optional
        Minimum number of knots (default: 6).
    max_knots : int, optional
        Maximum number of knots (default: 12).
    
    Returns
    -------
    knots : np.array
        Array of knot positions at data quantiles.
    """
    num_knots = determine_num_knots_adaptive(data, min_knots, max_knots)
    
    # Adjust num_knots to account for fixed first and last points
    num_interior_knots = num_knots - 2
    if num_interior_knots < 1:
        return np.array([np.min(data), np.max(data)])
    
    # Compute quantiles for interior knots
    quantiles = np.linspace(0, 1, num_interior_knots + 2)[1:-1]  # Exclude 0 and 1
    interior_knots = np.quantile(data, quantiles)
    
    # Add first and last points
    knots = np.concatenate(([np.min(data)], interior_knots, [np.max(data)]))
    return np.unique(knots)


def pcuf_adapted_smoother(x_data, weights, data_train, variable_name, utility_key,
                         min_knots=6, max_knots=12, linear_extrapolation=True):
    """
    PCUF adapted smoother using data-driven knot selection and monotonic splines.
    Ensures that the first and last data points are used as knots.
    
    Parameters
    ----------
    x_data : array-like
        Feature values for the specific variable.
    weights : dict
        Dictionary containing leaf values for utilities.
    data_train : pandas.DataFrame
        Training dataset.
    variable_name : str
        Name of the variable being smoothed.
    utility_key : str
        Key identifying the utility function.
    min_knots : int, optional
        Minimum number of knots (default: 6).
    max_knots : int, optional
        Maximum number of knots (default: 12).
    linear_extrapolation : bool, optional
        Whether to use linear extrapolation (default: True).
    
    Returns
    -------
    interpolator : function
        The fitted monotonic spline interpolator.
    """
    # Get knot positions
    x_knots = quantile_based_knots(x_data, min_knots, max_knots)
    
    # Get utility values at knot positions
    y_knots = []
    for x_knot in x_knots:
        # If this is the first or last knot, use the actual data value
        if x_knot == np.min(x_data) or x_knot == np.max(x_data):
            x_points, y_points = data_leaf_value(data_train[variable_name], weights[utility_key][variable_name], "data_weighted")
            # Find closest matching point
            idx = np.argmin(np.abs(np.array(x_points) - x_knot))
            y_knots.append(y_points[idx])
        else:
            # For interior knots, find the closest splitting point
            hist_values = weights[utility_key][variable_name]['Histogram values']
            split_points = weights[utility_key][variable_name]['Splitting points']
            idx = np.argmin(np.abs(np.array(split_points) - x_knot))
            y_knots.append(hist_values[idx])
    
    # Create interpolator using existing monotone_spline function
    x_spline = np.linspace(np.min(x_data), np.max(x_data), 1000)
    _, _, interpolator, _, _ = monotone_spline(
        x_spline=x_spline,
        weights=weights[utility_key][variable_name],
        x_knots=x_knots,
        y_knots=np.array(y_knots),
        linear_extrapolation=linear_extrapolation
    )
    
    # Store knots for plotting
    interpolator.x_knots = x_knots
    interpolator.y_knots = np.array(y_knots)
    
    return interpolator


def plot_pcuf_adapted(util_collection, utility_names, data_train, weights):
    """
    Plot PCUF adapted smoothed utility functions with raw data and knots.
    
    Parameters
    ----------
    util_collection : dict
        Dictionary containing smoothed interpolators for each utility and feature.
    utility_names : dict
        Mapping of utility keys to descriptive names.
    data_train : pandas.DataFrame
        Training dataset.
    weights : dict
        Dictionary containing weight data for each feature.
    """
    # Style settings matching original PCUF
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

    # Color mapping matching original PCUF style
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
            
            # Normalize raw data so that the first value is zero
            y_data_norm = y_data - y_data[0]
            ax.scatter(x_data, y_data_norm, color="k", s=4, alpha=1, 
                       edgecolors="none", label="Data")
            
            # Evaluate the smoother on a dense grid
            x_dense = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_smooth = util_collection[u][feature](x_dense)
            # Normalize the smoothed output so that its first value is zero
            y_smooth_norm = y_smooth - y_smooth[0]
            ax.plot(x_dense, y_smooth_norm, color=color_map.get(u, "#5badc7"),
                    linewidth=0.8, label="PCUF Adapted")
            
            # Add knots if available
            if hasattr(util_collection[u][feature], 'x_knots'):
                x_knots = util_collection[u][feature].x_knots
                y_knots = util_collection[u][feature].y_knots
                # Normalize knot utilities with the same offset as the raw data
                y_knots_norm = y_knots - y_data[0]
                ax.scatter(x_knots, y_knots_norm, color="#BE5C23", 
                           s=1, zorder=3, label="Knots")
            
            ax.set_xlabel(f"{feature} [km]" if "distance" in feature else feature)
            ax.set_ylabel(f"{utility_names.get(u, u)} utility")
            
            # Consolidate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            ax.grid(True)
            ax_idx += 1
    
    plt.tight_layout()
    plt.show()


# Hyperparameter optimization function

def pcuf_adapted_objective(params):
    """
    Objective function for optimizing min_knots and max_knots.
    """
    min_knots = params['min_knots']
    max_knots = params['max_knots']
    
    # Ensure max_knots > min_knots
    if max_knots <= min_knots:
        return {'loss': float('inf'), 'status': STATUS_OK}
    
    # Build the utility collection
    util_collection = {}
    smoothing_utilities = {
        '0': ['distance', 'dur_walking'],
        '1': ['distance', 'dur_cycling'],
        '2': ['dur_pt_rail', 'dur_pt_bus', 'cost_transit', 
              'dur_pt_int_waiting', 'dur_pt_int_walking', 'dur_pt_access'],
        '3': ['distance', 'dur_driving', 'cost_driving_fuel', 'driving_traffic_percent']
    }
    
    for u in smoothing_utilities:
        util_collection[u] = {}
        for var in smoothing_utilities[u]:
            try:
                feature_data = LPMC_train[var].dropna()
                if len(np.unique(feature_data)) >= 2:
                    util_collection[u][var] = pcuf_adapted_smoother(
                        feature_data, weights, LPMC_train, var, u,
                        min_knots=min_knots, max_knots=max_knots
                    )
                else:
                    continue
            except Exception as e:
                print(f"Error processing {var} for utility {u}: {e}")
                continue
    
    # Generate predictions and compute loss
    y_pred = smooth_predict(LPMC_test, util_collection)
    ce_loss = cross_entropy(y_pred, LPMC_test['choice'])
    
    return {'loss': ce_loss, 'status': STATUS_OK}


def optimize_pcuf_adapted():
    """
    Run hyperparameter optimization for PCUF adapted method.
    """
    param_space = {
        'min_knots': hp.uniformint('min_knots', low=3, high=10),
        'max_knots': hp.uniformint('max_knots', low=8, high=20),
    }
    
    trials = Trials()
    best_params = fmin(
        fn=pcuf_adapted_objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    
    print("Best PCUF Adapted Parameters:", best_params)
    return best_params