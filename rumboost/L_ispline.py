"""
I-Spline Smoothing Implementation for RUMBoost

This module implements I-spline smoothing as described in:
- De Leeuw, J. (2017). "Computing and Fitting Monotone Splines." Working Paper.

I-splines provide monotonic utility functions by constructing a basis of monotone
increasing functions and using non-positive coefficients to create monotone
decreasing curves. The implementation follows the approach of adding a constant
column for the intercept, keeping it unconstrained, while constraining all other
coefficients.
"""

__version__ = "L_ispline v_corrected_1.0"

import numpy as np
from sklearn.preprocessing import SplineTransformer
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt
import seaborn as sns
from rumboost.utility_smoothing import data_leaf_value

def i_spline_basis(x, knots, degree=3, debug=False):
    """
    Compute I-spline basis functions for given x and knots.
    This function uses a SplineTransformer to generate a B-spline basis
    (with include_bias=False), then integrates each column to produce
    an I-spline basis. Each column is normalized to [0,1].

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        The input data points.
    knots : array-like
        The knot positions (must be strictly increasing).
    degree : int, optional
        The spline degree (default is 3 for cubic).
    debug : bool, optional
        If True, prints diagnostic messages and optionally plots basis columns.

    Returns
    -------
    I_basis : np.ndarray, shape (n_samples, n_basis)
        The normalized I-spline basis functions.
    transformer : SplineTransformer
        The fitted transformer (to be reused for new x values).
    """
    x = np.array(x).reshape(-1, 1)
    
    # Ensure knots are properly formatted
    knots = np.array(knots)
    knots = np.sort(np.unique(knots))
    
    # Create B-spline basis with SplineTransformer
    transformer = SplineTransformer(degree=degree,
                                   knots=knots.reshape(-1, 1),
                                   include_bias=False)
    
    # Sort x for numerical stability during integration
    sort_idx = np.argsort(x.ravel())
    x_sorted = x[sort_idx]
    x_flat = x_sorted.ravel()
    
    # Transform to get B-spline basis
    B = transformer.fit_transform(x_sorted)
    n_samples, n_basis = B.shape
    
    # Integrate each B-spline to get I-spline basis
    I_basis = np.zeros_like(B)
    for j in range(n_basis):
        # Perform numerical integration
        I_col = np.concatenate(([0], cumulative_trapezoid(B[:, j], x_flat)))
        
        # Normalize to [0,1] range
        max_val = np.max(I_col)
        if max_val > 1e-10:  # Threshold to avoid numerical issues
            I_col = I_col / max_val
        I_basis[:, j] = I_col
    
    # Reorder back to original order
    I_basis_orig = np.zeros_like(I_basis)
    I_basis_orig[sort_idx] = I_basis
    
    if debug:
        print("[DEBUG] i_spline_basis:")
        print(f"  x range: {np.min(x)} to {np.max(x)} (n={len(x)})")
        print(f"  knots: {knots}")
        print(f"  B shape: {B.shape}, I_basis shape: {I_basis.shape}")
        for j in range(min(n_basis, 3)):
            print(f"    Col {j} range: {np.min(I_basis[:, j]):.4f} to {np.max(I_basis[:, j]):.4f}")
        
        if debug:
            plt.figure(figsize=(6, 4))
            for j in range(min(n_basis, 5)):
                plt.plot(x_flat, I_basis[:, j], label=f"I-col {j}")
            plt.title("I-Spline Basis (Normalized)")
            plt.xlabel("x")
            plt.ylabel("Basis value")
            plt.legend()
            plt.show()
    
    return I_basis_orig, transformer


def i_spline_smoother_decreasing(x, y, knots, degree=3,
                               endpoint_weight=0.5,
                               enforce_monotonicity=True,
                               debug=False):
    """
    Fit a monotone DECREASING I-spline smoother.
    
    Following the approach described in Ramsay (1988) and clarified in the splines2 package:
    1. Build I-spline basis
    2. Add a constant column (all ones) for the intercept
    3. Keep the intercept coefficient unconstrained
    4. Constrain all other coefficients to be non-positive for decreasing shape
    
    The final smoother is anchored to match the first data point.

    Parameters
    ----------
    x : array-like
        The input x-values.
    y : array-like
        The corresponding y-values.
    knots : array-like
        Knot positions.
    degree : int
        Spline degree (default=3).
    endpoint_weight : float
        Weight for soft endpoint matching (0 for no constraints).
    enforce_monotonicity : bool
        If True, apply constraints to enforce decreasing shape.
    debug : bool
        If True, prints debug info.

    Returns
    -------
    smoother : callable
        A function f(x_new) that returns the smoothed y-values.
    final_knots : np.ndarray
        The knot positions.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Sort the data for stable integration
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    
    # Build the I-spline basis
    I_basis, transformer = i_spline_basis(x_sorted, knots, degree=degree, debug=debug)
    n_samples, n_basis = I_basis.shape
    
    # Add a constant column (all ones) for the intercept
    H = np.column_stack((np.ones(n_samples), I_basis))
    
    # Small regularization to improve numerical stability
    reg_param = 1e-4
    n_params = H.shape[1]
    identity_reg = np.sqrt(reg_param) * np.eye(n_params)
    
    # Endpoint handling if enabled
    if endpoint_weight > 0:
        sqrt_w = np.sqrt(endpoint_weight)
        H_aug = np.vstack([
            H,
            sqrt_w * H[0:1, :],   # left endpoint
            sqrt_w * H[-1:, :],   # right endpoint
            identity_reg  # regularization
        ])
        y_aug = np.concatenate([
            y_sorted,
            [sqrt_w * y_sorted[0]],
            [sqrt_w * y_sorted[-1]],
            np.zeros(n_params)  # regularization target
        ])
    else:
        # Just add regularization
        H_aug = np.vstack([
            H,
            identity_reg
        ])
        y_aug = np.concatenate([
            y_sorted,
            np.zeros(n_params)
        ])
    
    if debug:
        print("[DEBUG] Smoother fit:")
        print(f"  H_aug shape: {H_aug.shape}, y_aug shape: {y_aug.shape}")
        print(f"  Condition number: {np.linalg.cond(H_aug):.2e}")
    
    # Bounds for monotone decreasing:
    # - First column is the intercept (unconstrained)
    # - All other columns (I-spline basis) have non-positive coefficients
    if enforce_monotonicity:
        lower = np.full(n_params, -np.inf)  # All parameters unbounded below
        upper = np.zeros(n_params)          # All parameters <= 0
        upper[0] = np.inf                   # Except intercept, which is free
        
        res = lsq_linear(H_aug, y_aug, bounds=(lower, upper))
        if not res.success:
            raise ValueError("lsq_linear failed: " + res.message)
        beta = res.x
        solver = "lsq_linear"
    else:
        # Unconstrained least squares
        beta, _, _, _ = np.linalg.lstsq(H_aug, y_aug, rcond=None)
        solver = "lstsq"
    
    if debug:
        print(f"[DEBUG] {solver} results:")
        print(f"  beta: {beta}")
        # Check if many coefficients are near zero
        near_zero = np.abs(beta) < 1e-10
        print(f"  Near-zero coefficients: {np.sum(near_zero)} out of {len(beta)}")
        print(f"  Residual norm: {np.linalg.norm(H @ beta - y_sorted):.4f}")
    
    # Anchor the curve so f(x_sorted[0]) == y_sorted[0]
    first_val_pred = H[0, :] @ beta
    anchor_shift = y_sorted[0] - first_val_pred
    
    def smoother(x_new):
        """Predict using the fitted I-spline model"""
        x_new = np.asarray(x_new).reshape(-1, 1)
        
        # Clamp to training range for numerical stability
        x_min, x_max = np.min(x_sorted), np.max(x_sorted)
        x_new_clipped = np.clip(x_new, x_min, x_max)
        
        # Transform to B-spline basis
        B_new = transformer.transform(x_new_clipped)
        
        # Convert to I-spline basis
        n_new = len(x_new_clipped)
        I_new = np.zeros((n_new, n_basis))
        
        # Sort for stable integration
        sort_idx_new = np.argsort(x_new_clipped.ravel())
        x_new_sorted = x_new_clipped[sort_idx_new]
        B_new_sorted = B_new[sort_idx_new]
        
        # Integrate and normalize
        for j in range(n_basis):
            I_col = np.concatenate(([0], cumulative_trapezoid(B_new_sorted[:, j], x_new_sorted.ravel())))
            max_val = np.max(I_col)
            if max_val > 1e-10:
                I_col = I_col / max_val
            
            # Restore original order
            I_new_j = np.zeros(n_new)
            I_new_j[sort_idx_new] = I_col
            I_new[:, j] = I_new_j
        
        # Add constant column
        H_new = np.column_stack((np.ones(n_new), I_new))
        
        # Apply model and anchor shift
        return H_new @ beta + anchor_shift
    
    return smoother, knots


def plot_ispline_smoothing(util_collection, utility_names, data_train, weights):
    """
    Plot the I-spline smoothed utility functions alongside raw data.
    Each plot is rendered in its own figure.
    
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

    color_map = {
        "walking": "b",
        "cycling": "r",
        "rail": "g",
        "driving": "orange",
        "swissmetro": "#6b8ba4"
    }

    for u in sorted(util_collection.keys()):
        for feature in util_collection[u]:
            plt.figure(figsize=(3.49, 2.09), dpi=1000)
            ax = plt.gca()

            try:
                x_data, y_data = data_leaf_value(data_train[feature],
                                                 weights[u][feature],
                                                 technique="data_weighted")
            except Exception as e:
                print(f"[ERROR] Utility {u}, feature {feature}: {e}")
                continue

            y_data_norm = y_data - y_data[0]

            ax.scatter(x_data, y_data_norm, color="k", s=4, alpha=1,
                       edgecolors="none", label="Data")

            x_dense = np.linspace(np.min(x_data), np.max(x_data), 200)
            y_smooth = util_collection[u][feature](x_dense)
            y_smooth_norm = y_smooth - y_smooth[0]

            ax.plot(x_dense, y_smooth_norm,
                    color=color_map.get(u, "#5badc7"),
                    linewidth=0.8, label="I-Spline")

            ax.set_xlabel(f"{feature} [km]" if "distance" in feature else feature)
            ax.set_ylabel(f"{utility_names.get(u, u)} utility")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()