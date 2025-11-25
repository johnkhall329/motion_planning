"""
PathPlanning/trajectory.py

Milestone 1: smoothing, arc-length resampling, heading normalization, curvature estimation.
Stage A: time-parameterize path with an analytic trapezoidal (or triangular) velocity profile.

Functions:
- path_pixels_to_meters(path_px, meters_per_pixel, y_inverted=True)
- unwrap_headings(headings)
- smooth_and_resample(path_px, spacing_m=0.1, meters_per_pixel=0.01, smoothing=True)
- compute_curvature(x, y)
- generate_trapezoidal_profile(s_total, v0, vf, v_max, a_max, d_max, dt=0.02)
- parameterize_path_trapezoid(resampled_path, v0, vf, v_max, a_max, d_max, dt=0.02)
- quick_visual_check(...)  # optional plotting (matplotlib)
"""

from typing import Tuple, Optional
import numpy as np
import time
import os
import pandas as pd

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
    Resample the polyline (x,y) at positions s_new along cumulative arc length using linear interpolation.
    If headings provided, they will be interpolated (after unwrapping).
    """
    s = cumulative_arc_length(x, y)
    # clamp s_new range
    s_min, s_max = s[0], s[-1]
    s_new = np.clip(s_new, s_min, s_max)

    # linear interpolation fallback (numpy.interp requires increasing x)
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
        smoothing_factor = 0.01
        cs_x = UnivariateSpline(s, x, s=smoothing_factor)
        cs_y = UnivariateSpline(s, y, s=smoothing_factor)

        # evaluate dense points then resample
        s_dense = np.linspace(s[0], s[-1], max(200, len(s) * 10))
        x_dense = cs_x(s_dense)
        y_dense = cs_y(s_dense)
        # headings: derivative-based
        dx_ds = cs_x.derivative()(s_dense)
        dy_ds = cs_y.derivative()(s_dense)
        # heading convention: x right, y up
        # atan2(dy, dx) gives angle from +x axis; subtract pi/2 so that angle=0 corresponds to +y (up)
        h_dense = np.arctan2(dy_ds, dx_ds) - (np.pi / 2.0)
        # Now create uniform s_new spacing
        s_new = np.arange(s_dense[0], s_dense[-1] + 1e-12, spacing_m)
        # Interpolate dense to s_new
        x_new = np.interp(s_new, s_dense, x_dense)
        y_new = np.interp(s_new, s_dense, y_dense)
        h_new = np.interp(s_new, s_dense, h_dense)
    else:
        # no scipy: do simple resample from original polyline using linear interp
        s = cumulative_arc_length(x, y)
        if s[-1] < spacing_m or len(x) < 2:
            # path too short or trivial, just return original but converted (ensure headings unwrapped)
            return np.vstack([x, y, unwrap_headings(h)]).T
        s_new = np.arange(0.0, s[-1] + 1e-12, spacing_m)
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
    if n < 3:
        return np.zeros(n)

    # approximate derivatives using central differences where possible
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


# ---- Stage A: trapezoidal / triangular velocity profile generation ----
def generate_trapezoidal_profile(s_total: float,
                                 v0: float,
                                 vf: float,
                                 v_max: float,
                                 a_max: float,
                                 d_max: float,
                                 dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytic trapezoidal (or triangular) velocity profile generator along a 1D path.

    Inputs:
        s_total: total path length (meters)
        v0: initial velocity (m/s), >= 0
        vf: final velocity (m/s), >= 0
        v_max: desired max velocity (m/s), > 0
        a_max: max acceleration (m/s^2) during accel phase, > 0
        d_max: max deceleration (m/s^2) during decel phase, > 0 (positive number)
        dt: time step for sampling the profile (seconds)

    Returns:
        t: (N,) timestamps from 0 to T
        s_traj: (N,) cumulative distance along path at each timestamp
        v_traj: (N,) velocity along path at each timestamp

    Notes:
        - Handles non-zero v0 and vf.
        - If v_max is not reachable given s_total and accel/decel limits, returns a triangular profile with peak vm.
        - Robust to small numerical issues.
    """
    # sanity clamp inputs
    v0 = max(0.0, float(v0))
    vf = max(0.0, float(vf))
    v_max = max(1e-9, float(v_max))
    a_max = max(1e-9, float(a_max))
    d_max = max(1e-9, float(d_max))
    s_total = max(0.0, float(s_total))

    # trivial zero-length path
    if s_total <= EPS:
        t = np.array([0.0])
        s_traj = np.array([0.0])
        v_traj = np.array([v0])
        return t, s_traj, v_traj

    # distance needed to accelerate from v0 to v (generic):
    # s = (v^2 - v0^2) / (2*a)
    # Compute distances for accel to v_max and decel from v_max to vf
    s_acc_to_vmax = 0.0
    if v_max > v0:
        s_acc_to_vmax = (v_max * v_max - v0 * v0) / (2.0 * a_max)
    else:
        s_acc_to_vmax = 0.0

    s_dec_from_vmax = 0.0
    if v_max > vf:
        s_dec_from_vmax = (v_max * v_max - vf * vf) / (2.0 * d_max)
    else:
        s_dec_from_vmax = 0.0

    # check if we can fit full trapezoid (accel to v_max, cruise, decel)
    if s_acc_to_vmax + s_dec_from_vmax <= s_total:
        # trapezoidal case: compute times
        t_acc = (max(v_max, v0) - v0) / a_max if v_max > v0 else 0.0
        t_dec = (max(v_max, vf) - vf) / d_max if v_max > vf else 0.0
        s_cruise = s_total - s_acc_to_vmax - s_dec_from_vmax
        t_cruise = s_cruise / v_max if v_max > 0 else 0.0
        T = t_acc + t_cruise + t_dec

        # build time vector
        t = np.arange(0.0, T + dt * 0.5, dt)

        # piecewise compute v(t)
        v_traj = np.zeros_like(t)
        s_traj = np.zeros_like(t)

        # time boundaries
        t1 = t_acc
        t2 = t_acc + t_cruise  # start of decel
        for i, ti in enumerate(t):
            if ti <= t1:
                # accel from v0 with a_max
                v = v0 + a_max * ti
                s = v0 * ti + 0.5 * a_max * ti * ti
            elif ti <= t2:
                # cruise at v_max
                v = v_max
                # distance covered until t1
                s_at_t1 = v0 * t1 + 0.5 * a_max * t1 * t1
                s = s_at_t1 + v_max * (ti - t1)
            else:
                # decel phase
                td = ti - t2
                # distance at start of decel
                s_at_t2 = v0 * t1 + 0.5 * a_max * t1 * t1 + v_max * (t2 - t1)
                # decel from v_max with -d_max
                v = v_max - d_max * td
                # guard negative due to numerical
                if v < 0.0:
                    v = 0.0
                s = s_at_t2 + v_max * td - 0.5 * d_max * td * td
            v_traj[i] = v
            s_traj[i] = s
        # numeric clamp final s to s_total and final v to vf
        s_traj[-1] = s_total
        v_traj[-1] = vf
        return t, s_traj, v_traj

    # Else triangular profile: peak velocity vm < v_max
    # Solve for vm^2 using energy/distance equations:
    # s_total = (vm^2 - v0^2)/(2*a_max) + (vm^2 - vf^2)/(2*d_max)
    A = 1.0 / (2.0 * a_max) + 1.0 / (2.0 * d_max)
    rhs = s_total + (v0 * v0) / (2.0 * a_max) + (vf * vf) / (2.0 * d_max)
    vm2 = rhs / A
    # Numerical safety
    if vm2 < 0.0:
        vm2 = 0.0
    vm = np.sqrt(vm2)

    # times
    t_acc = max(0.0, (vm - v0) / a_max)
    t_dec = max(0.0, (vm - vf) / d_max)
    T = t_acc + t_dec

    t = np.arange(0.0, T + dt * 0.5, dt)
    v_traj = np.zeros_like(t)
    s_traj = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= t_acc:
            v = v0 + a_max * ti
            s = v0 * ti + 0.5 * a_max * ti * ti
        else:
            td = ti - t_acc
            # deceleration from vm down to vf
            v = vm - d_max * td
            if v < 0.0:
                v = 0.0
            # s at end of accel (t_acc)
            s_at_tacc = v0 * t_acc + 0.5 * a_max * t_acc * t_acc
            s = s_at_tacc + vm * td - 0.5 * d_max * td * td
        v_traj[i] = v
        s_traj[i] = s
    # numeric clamp
    s_traj[-1] = s_total
    v_traj[-1] = vf
    return t, s_traj, v_traj


def parameterize_path_trapezoid(resampled_path: np.ndarray,
                                v0: float,
                                vf: float,
                                v_max: float,
                                a_max: float,
                                d_max: float,
                                dt: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a resampled path (M,3) from smooth_and_resample -> columns x,y,heading,
    produce a time-parameterized trajectory using a trapezoidal/triangular velocity profile.

    Inputs:
        resampled_path: (M,3) array with columns [x (m), y (m), heading (rad)]
        v0: initial path speed (m/s)
        vf: final path speed (m/s)
        v_max: max cruising speed (m/s)
        a_max: accel (m/s^2)
        d_max: decel (m/s^2)
        dt: sample timestep for output (s)

    Returns:
        traj: (N,6) array with columns:
            [t, x_ref, y_ref, heading_ref, s_ref, v_ref]
        times: (N,) same as traj[:,0] (for convenience)
    """
    resampled = np.asarray(resampled_path)
    assert resampled.ndim == 2 and resampled.shape[1] >= 2

    x = resampled[:, 0]
    y = resampled[:, 1]
    headings = resampled[:, 2] if resampled.shape[1] > 2 else np.zeros_like(x)

    # cumulative s of the resampled path
    s = cumulative_arc_length(x, y)
    s_total = s[-1]

    # generate profile in s-space
    t, s_traj, v_traj = generate_trapezoidal_profile(s_total, v0, vf, v_max, a_max, d_max, dt=dt)

    # map s_traj -> x,y,heading using linear interpolation on s
    if len(s) < 2:
        # degenerate: single point
        x_ref = np.full_like(s_traj, x[0])
        y_ref = np.full_like(s_traj, y[0])
        h_ref = np.full_like(s_traj, headings[0] if len(headings) > 0 else 0.0)
    else:
        # for safety, allow s to end exactly at s_total
        s_for_interp = s
        x_ref = np.interp(s_traj, s_for_interp, x)
        y_ref = np.interp(s_traj, s_for_interp, y)
        # unwrap headings first to avoid wrap issues
        h_unwrapped = unwrap_headings(headings)
        h_ref_unwrapped = np.interp(s_traj, s_for_interp, h_unwrapped)
        h_ref = (h_ref_unwrapped + np.pi) % (2.0 * np.pi) - np.pi

    # assemble trajectory: [t, x, y, heading, s, v]
    traj = np.vstack([t, x_ref, y_ref, h_ref, s_traj, v_traj]).T
    return traj, t


# ---- Example helper for quick checks + plotting ----
def quick_visual_check(original_path_px: np.ndarray,
                       resampled_m: np.ndarray,
                       traj: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (10, 6),
                       show: bool = True):
    """
    If matplotlib available, plot original vs resampled path and optionally the time-parameterized profile.
    Usage:
        quick_visual_check(path_px, resampled, traj)
    where traj is the output from parameterize_path_trapezoid (N,6).
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping visual check.")
        return

    orig_m = path_pixels_to_meters(original_path_px)
    x0, y0 = orig_m[:, 0], orig_m[:, 1]
    xr, yr = resampled_m[:, 0], resampled_m[:, 1]

    fig = plt.figure(figsize=figsize)
    ax_path = fig.add_subplot(1, 2, 1)
    ax_profile = fig.add_subplot(2, 2, 2)
    ax_speed = fig.add_subplot(2, 2, 4)

    # Path plot
    ax_path.plot(x0, y0, '.', label='original (converted)')
    ax_path.plot(xr, yr, 'o-', label='smoothed/resampled', markersize=3)
    ax_path.set_title('Path (meters)')
    ax_path.axis('equal')
    ax_path.set_xlabel('x (m)')
    ax_path.set_ylabel('y (m)')
    ax_path.legend()

    if traj is not None and traj.shape[1] >= 6:
        t = traj[:, 0]
        s = traj[:, 4]
        v = traj[:, 5]
        x_ref = traj[:, 1]
        y_ref = traj[:, 2]
        # overlay reference points on path
        ax_path.plot(x_ref, y_ref, 'x', label='time-param points', markersize=4)
        # s vs t
        ax_profile.plot(t, s, '-')
        ax_profile.set_title('s(t)')
        ax_profile.set_xlabel('t (s)')
        ax_profile.set_ylabel('s (m)')
        # v vs t
        ax_speed.plot(t, v, '-')
        ax_speed.set_title('v(t)')
        ax_speed.set_xlabel('t (s)')
        ax_speed.set_ylabel('v (m/s)')

    fig.tight_layout()
    if show:
        plt.show()


# ---- small unit tests you can run from testing.py ----
def _test_circle_curvature():
    # radius = 5 m, sample points on circle, curvature should be ~ 1/5 = 0.2
    R = 5.0
    thetas = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    x = R * np.cos(thetas)
    y = R * np.sin(thetas)
    kappa = compute_curvature(x, y)
    # ignore boundary few points
    mean_kappa = np.mean(np.abs(kappa[5:-5]))
    print("Expected curvature:", 1.0 / R, "mean abs curvature:", mean_kappa)


def _test_trapezoid():
    # simple straight path of 10 m
    s_total = 10.0
    v0 = 0.5
    vf = 0.5
    v_max = 2.0
    a_max = 1.0
    d_max = 1.0
    t, s_traj, v_traj = generate_trapezoidal_profile(s_total, v0, vf, v_max, a_max, d_max, dt=0.01)
    print("T_total:", t[-1], "s_end:", s_traj[-1], "v_end:", v_traj[-1])
    # quick checks
    assert abs(s_traj[-1] - s_total) < 1e-3
    assert abs(v_traj[-1] - vf) < 1e-3

def animate_trajectory(traj: np.ndarray):
    """
    Animate the trajectory point by point in real time.
    traj: (N,6) array [t, x, y, heading, s, v]
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping animation.")
        return

    if traj is None or len(traj) < 2:
        print("Trajectory too short to animate.")
        return

    x = traj[:, 1]
    y = traj[:, 2]
    t = traj[:, 0]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Trajectory Animation")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis('equal')

    # Plot full path faintly
    ax.plot(x, y, '--', color='gray', alpha=0.3)

    point, = ax.plot([], [], 'ro', markersize=6)  # moving point

    for i in range(len(x)):
        point.set_data([x[i]], [y[i]])
        fig.canvas.draw()
        fig.canvas.flush_events()
        if i < len(t) - 1:
            dt = t[i + 1] - t[i]
            plt.pause(dt)  # pause for the time difference

    plt.ioff()
    plt.show()

import os
import numpy as np
import pandas as pd

def save_traj_to_csv(
        traj: np.ndarray,
        filename: str = 'traj.csv',
    ) -> str:
    """
    Save a trajectory ndarray to CSV using pandas.

    Parameters:
      traj: (N,6) or (N,M) ndarray. Expected standard columns:
        [t, x_ref, y_ref, heading_ref, s_ref, v_ref]
      filename: output file path (will create parent directories)

    Returns:
      The path to the saved file.
    """
    if traj is None:
        raise ValueError("traj is None")

    arr = np.asarray(traj)
    if arr.ndim != 2:
        raise ValueError("traj must be a 2D ndarray")

    # Ensure directory exists
    outdir = os.path.dirname(os.path.abspath(filename))
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Build header according to number of columns
    ncols = arr.shape[1]
    if ncols >= 6:
        cols = ['t', 'x_ref', 'y_ref', 'heading_ref', 's_ref', 'v_ref'] + \
               [f'c{i}' for i in range(6, ncols)]
    else:
        cols = [f'c{i}' for i in range(ncols)]

    df = pd.DataFrame(arr, columns=cols)

    # Save with pandas
    df.to_csv(filename, index=False, float_format='%.3f')
    print(f"Trajectory saved to {filename}.")

    return filename



if __name__ == "__main__":
    # quick inline smoke test - expects hybrid_astar_path.npy to exist if you want to run full demo
    # run the trapezoid unit test
    _test_trapezoid()

    # If a hybrid path file exists, demonstrate the pipeline
    try:
        filepath = 'hybrid_astar_path.npy'
        hybrid_path = np.load(filepath)
        resampled = smooth_and_resample(hybrid_path, spacing_m=0.1)
        traj, times = parameterize_path_trapezoid(resampled,
                                                  v0=4,
                                                  vf=4,
                                                  v_max=8.0,
                                                  a_max=2.0,
                                                  d_max=2.0,
                                                  dt=0.02)
        quick_visual_check(hybrid_path, resampled, traj)
    except FileNotFoundError:
        print("No example hybrid_astar_path.npy found; skip pipeline demo.")

    # print("len original: ", len(hybrid_path))
    # print("len resampled: ", len(resampled))
    # print("len traj: ", len(traj))

    import math
    # for item in traj:
    #     print(math.degrees(item[3]))

    # animate_trajectory(traj)
    save_traj_to_csv(traj)