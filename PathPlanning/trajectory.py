"""
PathPlanning/trajectory.py

Milestone 1: smoothing, arc-length resampling, heading normalization, curvature estimation.

Functions:
- path_pixels_to_meters(path_px, meters_per_pixel, y_inverted=True)
- unwrap_headings(headings)
- smooth_and_resample(path_px, spacing_m=0.1, meters_per_pixel=0.01, smoothing=True)
- compute_curvature(x, y)
- quick_visual_check(...)  # optional plotting (matplotlib)
"""

from typing import Tuple, List, Optional
import numpy as np
try:
    from Kinematics.parameters import *
except ModuleNotFoundError:
    METERS_PER_PIXEL = 0.035

# Optional: SciPy for robust cubic spline interpolation if available.
try:
    from scipy.interpolate import UnivariateSpline
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

EPS = 1e-9


def path_pixels_to_meters(path_px: np.ndarray,
                          meters_per_pixel: float = METERS_PER_PIXEL,
                          y_inverted: bool = True) -> np.ndarray:
    """
    Convert path array from pixels to meters and fix y-axis sign.
    Input:
        path_px: (N,3) array of (x_pix, y_pix, heading_rad)
        meters_per_pixel: conversion factor
        y_inverted: if True, convert y_pix (down) -> y_m (up) via y_m = -y_pix * mpp
    Returns:
        path_m: (N,3) array of (x_m, y_m, heading_rad)
    Notes:
        Does NOT change headings except sign flip if y_inverted (heading definition depends on src).
    """
    path_px = np.asarray(path_px)
    assert path_px.ndim == 2 and path_px.shape[1] >= 2
    x_px = path_px[:, 0].astype(float)
    y_px = path_px[:, 1].astype(float)
    headings = path_px[:, 2] if path_px.shape[1] > 2 else np.zeros_like(x_px)

    x_m = x_px * meters_per_pixel
    y_m = (-y_px if y_inverted else y_px) * meters_per_pixel
    # If headings in source assume y-down, flip sign of heading if necessary:
    if y_inverted:
        headings = -headings

    out = np.vstack([x_m, y_m, headings]).T
    return out


def unwrap_headings(headings: np.ndarray) -> np.ndarray:
    """
    Unwrap heading sequence to avoid ±pi jumps (continuous progression).
    """
    return np.unwrap(headings)


def cumulative_arc_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return cumulative arc-length array s (same length as x,y).
    """
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx * dx + dy * dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    return s


def resample_by_arclength(x: np.ndarray, y: np.ndarray, s_new: np.ndarray,
                          headings: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Resample the polyline (x,y) at positions s_new along cumulative arc length using linear or spline interpolation.
    If headings provided, they will be interpolated (after unwrapping).
    """
    s = cumulative_arc_length(x, y)
    # clamp s_new range
    s_min, s_max = s[0], s[-1]
    s_new = np.clip(s_new, s_min, s_max)

    # linear interpolation fallback
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    h_new = None
    if headings is not None:
        h_unwrapped = unwrap_headings(headings)
        h_new = np.interp(s_new, s, h_unwrapped)
        # re-wrap to (-pi, pi]
        h_new = (h_new + np.pi) % (2 * np.pi) - np.pi

    return x_new, y_new, h_new


def smooth_and_resample(path_px: np.ndarray,
                        spacing_m: float = 0.1,
                        meters_per_pixel: float = METERS_PER_PIXEL,
                        smoothing: bool = True) -> np.ndarray:
    """
    Top-level helper:
    - Convert path px -> meters (assumes y down in pixel coordinates)
    - Optionally smooth with cubic spline (if scipy available)
    - Resample uniformly by arc length at spacing_m
    Returns:
        resampled: (M,3) array of (x_m, y_m, heading_rad)
    """
    path_m = path_pixels_to_meters(path_px, meters_per_pixel, y_inverted=True)
    x = path_m[:, 0]
    y = path_m[:, 1]
    h = path_m[:, 2] if path_m.shape[1] > 2 else np.zeros_like(x)

    if smoothing and SCIPY_AVAILABLE and len(x) >= 4:
        s = cumulative_arc_length(x, y)
        # parametric spline for x(s), y(s)

        smoothing_factor = 0.001

        cs_x = UnivariateSpline(s, x, s=smoothing_factor)
        cs_y = UnivariateSpline(s, y, s=smoothing_factor)

        # evaluate dense points then resample
        s_dense = np.linspace(s[0], s[-1], max(200, len(s)*10))
        x_dense = cs_x(s_dense)
        y_dense = cs_y(s_dense)
        # headings: derivative-based
        dx_ds = cs_x.derivative()(s_dense)
        dy_ds = cs_y.derivative()(s_dense)
        # heading convention: x right, y up (we converted y to meters with y positive up)
        # desired heading: 0 = UP (negative screen-y), positive CCW
        # atan2(dy, dx) gives angle from +x axis; subtract pi/2 so that angle=0 corresponds to +y (up)
        h_dense = np.arctan2(dy_ds, dx_ds) - (np.pi / 2.0)
        # Now create uniform s_new spacing
        s_new = np.arange(s_dense[0], s_dense[-1], spacing_m)
        # Interpolate dense to s_new
        x_new = np.interp(s_new, s_dense, x_dense)
        y_new = np.interp(s_new, s_dense, y_dense)
        h_new = np.interp(s_new, s_dense, h_dense)
    else:
        # no scipy: do simple resample from original polyline using linear interp
        s = cumulative_arc_length(x, y)
        if s[-1] < spacing_m:
            # path too short, just return original but converted
            return path_m
        s_new = np.arange(0.0, s[-1], spacing_m)
        x_new, y_new, h_new = resample_by_arclength(x, y, s_new, headings=h)

    resampled = np.vstack([x_new, y_new, unwrap_headings(h_new)]).T
    return resampled


def compute_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute curvature kappa(s) for a parametric curve (x(s), y(s)) given points sampled uniformly in s (or nearly).
    Use central finite differences for derivatives.

    Returns kappa array of same length (nan or 0 near boundaries).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    kappa = np.zeros(n)

    # approximate derivatives using central differences where possible
    # assume uniform spacing ds; compute ds per-sample for robustness
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx * dx + dy * dy) ** 1.5
    # protect small denom
    denom_safe = np.where(denom < EPS, np.inf, denom)
    kappa = (dx * ddy - dy * ddx) / denom_safe
    # clamp extreme values
    kappa = np.nan_to_num(kappa, nan=0.0, posinf=0.0, neginf=0.0)
    return kappa


# ---- Example helper for quick checks ----
def quick_visual_check(original_path_px: np.ndarray,
                       resampled_m: np.ndarray,
                       figsize: Tuple[int, int] = (8, 8),
                       show: bool = True):
    """
    If matplotlib is available, plot original vs resampled (converted to meters).
    Usage: quick_visual_check(path_from_hybrid, resampled) where resampled is returned by smooth_and_resample.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping visual check.")
        return

    orig_m = path_pixels_to_meters(original_path_px)
    x0, y0 = orig_m[:, 0], orig_m[:, 1]
    xr, yr = resampled_m[:, 0], resampled_m[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: original path (converted to meters) - markers only
    axes[0].plot(x0, y0, '.', color='C0')
    axes[0].set_title('Original (converted)')
    axes[0].axis('equal')

    # Right: smoothed/resampled path - markers only
    axes[1].plot(xr, yr, 'o', color='C1')
    axes[1].set_title('Smoothed / Resampled')
    axes[1].axis('equal')

    # shared decorations
    for ax in axes:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    fig.suptitle('Path: original vs resampled')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()


# ---- small unit tests you can run from testing.py ----
def _test_circle_curvature():
    # radius = 5 m, sample points on circle, curvature should be ~ 1/5 = 0.2
    R = 5.0
    thetas = np.linspace(0, 2*np.pi, 200, endpoint=False)
    x = R * np.cos(thetas)
    y = R * np.sin(thetas)
    kappa = compute_curvature(x, y)
    # ignore boundary few points
    mean_kappa = np.mean(np.abs(kappa[5:-5]))
    print("Expected curvature:", 1.0/R, "mean abs curvature:", mean_kappa)


if __name__ == "__main__":
    # quick inline smoke test
    # _test_circle_curvature()

    filepath = 'hybrid_astar_path.npy'
    hybrid_path = np.load(filepath)

    for loc in hybrid_path:
        x, y, phi = loc
        print(f"x: {x:.2f} pix, y: {y:.2f} pix, heading: {phi*180/np.pi:.1f} deg")

    resampled = smooth_and_resample(hybrid_path, spacing_m=0.3)
    
    for loc in resampled:
        x, y, phi = loc
        print(f"x: {x:.2f} m, y: {y:.2f} m, heading: {phi*180/np.pi:.1f} deg")
    
    quick_visual_check(hybrid_path, resampled)