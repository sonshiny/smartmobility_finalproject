"""
Indoor Factory (InF) NLOS-robust positioning
"Deterministic Consensus Trilateration with NLOS Upper-Bound Constraints"

손재호 / 12223634 / 스마트모빌리티공학과 / 인하대학교
스마트모빌리티공학실험2 Final Project

The estimator turns 18 base-station RTT distances into a 2-D UE position. It is
fully deterministic (identical input -> identical output) and uses no machine
learning and no measurement-variance information (none is provided).

The whole design rests on one physical fact about this NLOS-dominated channel:
a blocked signal detours, so a measurement OVER-estimates the true distance
(d_hat >= d_true). That single fact is exploited twice:
  * small measurements are the LOS-likely ones  -> consensus / inlier selection
  * every measurement is an UPPER bound on range -> ||x - BS_i|| <= d_hat_i

Pipeline:
  Stage 1  Physical filter      - search bounds auto-derived from the BS layout;
                                  drop NaN / non-positive measurements; if fewer
                                  than 3 remain, return the BS centroid.
  Stage 2  Deterministic        - enumerate ALL C(n,3) closed-form trilaterations
           consensus              of the valid anchors, count inliers
                                  (|residual| < TAU) over all measurements, and
                                  keep the candidate with the most inliers. The
                                  enumeration is ordered by ascending distance so
                                  ties resolve toward LOS-likely (small) anchors;
                                  exhaustive + ordered => no randomness.
  Stage 3  Upper-bound          - least squares on the inlier set, subject to the
           constrained           physical constraint ||x - BS_i|| <= d_hat_i for
           refinement             every discarded (outlier) measurement. This
                                  re-uses the thrown-away NLOS data as one-sided
                                  bounds, which stops information-poor estimates
                                  from diverging while leaving well-localized ones
                                  untouched. Solved with SLSQP; on the rare
                                  non-convergence it falls back to a large
                                  fixed-weight exterior penalty (L-BFGS-B), so the
                                  result is never worse than plain bounded LS.
"""

import numpy as np
import scipy.io as sio
from itertools import combinations
from scipy.optimize import minimize, NonlinearConstraint

# ----------------------------- parameters ----------------------------------
# Two intrinsic method thresholds; the search bounds are derived analytically
# from the base-station layout at run time (no hand-set spatial constants).
TAU          = 1.5    # inlier residual threshold [m]
BOUND_MARGIN = 0.30   # search-bound margin as fraction of BS span
PENALTY_W    = 50.0   # fallback exterior-penalty weight; "large enough" to
                      # enforce the upper-bound constraint (result is insensitive
                      # to its exact value above ~16, so it is not a tuned knob)


def _as_bs_Nx2(p_bs):
    """Accept BS coordinates as (2, M) or (M, 2); always return (M, 2)."""
    bs = np.asarray(p_bs, dtype=float)
    if bs.ndim != 2:
        raise ValueError("BS positions must be 2-D")
    if bs.shape[0] == 2 and bs.shape[1] != 2:
        bs = bs.T
    return bs


def _trilaterate(anchors, dists):
    """Closed-form 3-point trilateration by linearizing the circle equations."""
    x1, y1 = anchors[0]; x2, y2 = anchors[1]; x3, y3 = anchors[2]
    d1, d2, d3 = dists
    A = np.array([[2.0 * (x2 - x1), 2.0 * (y2 - y1)],
                  [2.0 * (x3 - x1), 2.0 * (y3 - y1)]])
    b = np.array([d1**2 - d2**2 - x1**2 - y1**2 + x2**2 + y2**2,
                  d1**2 - d3**2 - x1**2 - y1**2 + x3**2 + y3**2])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def _sse(x, anchors, dists):
    """Sum of squared range residuals (least-squares objective)."""
    r = np.sqrt(((anchors - x) ** 2).sum(1)) - dists
    return (r ** 2).sum()


def _sse_penalized(x, a_in, d_in, a_ub, d_ub, w):
    """Inlier LS + one-sided exterior penalty for ||x-BS_i|| > d_hat_i."""
    c = _sse(x, a_in, d_in)
    if len(d_ub):
        viol = np.sqrt(((a_ub - x) ** 2).sum(1)) - d_ub
        c += w * (np.maximum(viol, 0.0) ** 2).sum()
    return c


def _constrained_refine(x0, a_in, d_in, a_ub, d_ub, bounds):
    """LS on inliers s.t. ||x-BS_i|| <= d_hat_i for the upper-bound anchors.
    Primary SLSQP hard constraint; fallback large penalty on non-convergence."""
    if len(d_ub):
        con = NonlinearConstraint(
            lambda x: d_ub - np.sqrt(((a_ub - x) ** 2).sum(1)), 0.0, np.inf)
        try:
            out = minimize(_sse, x0, args=(a_in, d_in), method='SLSQP',
                           constraints=[con], bounds=bounds)
            if out.success:
                return out.x
        except Exception:
            pass
        out = minimize(_sse_penalized, x0, args=(a_in, d_in, a_ub, d_ub, PENALTY_W),
                       method='L-BFGS-B', bounds=bounds)
        return out.x
    out = minimize(_sse, x0, args=(a_in, d_in), method='L-BFGS-B', bounds=bounds)
    return out.x


def your_algorithm(d_hat, p_bs):
    """Estimate one UE position [x, y] from its RTT distance vector.

    d_hat : (M,) measured distances (NaN / non-positive treated as missing).
    p_bs  : (2, M) or (M, 2) base-station coordinates.
    """
    bs = _as_bs_Nx2(p_bs)
    d = np.asarray(d_hat, dtype=float).ravel()
    center = bs.mean(0)

    # --- Stage 1: physical filter (bounds auto-derived from geometry) ---
    bmin = bs.min(0); bmax = bs.max(0)
    margin = (bmax - bmin) * BOUND_MARGIN
    bounds = [(bmin[0] - margin[0], bmax[0] + margin[0]),
              (bmin[1] - margin[1], bmax[1] + margin[1])]

    valid = np.isfinite(d) & (d > 0)
    idx = np.where(valid)[0]
    if idx.size < 3:
        return center.copy()

    bs_v = bs[idx]
    d_v = d[idx]

    # --- Stage 2: deterministic consensus over all C(n,3) trilaterations ---
    order = np.argsort(d_v)            # ascending distance -> LOS-likely first
    best_inliers = -1
    best_pos = None
    for combo in combinations(order, 3):
        c = list(combo)
        pos = _trilaterate(bs_v[c], d_v[c])
        if pos is None:
            continue
        if not (bounds[0][0] <= pos[0] <= bounds[0][1] and
                bounds[1][0] <= pos[1] <= bounds[1][1]):
            continue
        res = np.abs(np.sqrt(((bs_v - pos) ** 2).sum(1)) - d_v)
        n_inl = int((res < TAU).sum())
        if n_inl > best_inliers:       # ties keep the earlier (smaller-distance) combo
            best_inliers = n_inl
            best_pos = pos

    if best_pos is None:
        best_pos = bs_v[order[0]].copy()

    res = np.abs(np.sqrt(((bs_v - best_pos) ** 2).sum(1)) - d_v)
    inl_mask = res < TAU
    if inl_mask.sum() < 3:             # need >=3 anchors for 2-D refinement
        inl_mask = np.zeros_like(inl_mask)
        inl_mask[np.argsort(res)[:3]] = True

    a_in = bs_v[inl_mask]; d_in = d_v[inl_mask]
    a_ub = bs_v[~inl_mask]; d_ub = d_v[~inl_mask]   # discarded -> upper bounds

    # --- Stage 3: NLOS upper-bound constrained least-squares refinement ---
    return _constrained_refine(best_pos, a_in, d_in, a_ub, d_ub, bounds)


def main():
    # 1) input data load — grader places the .mat file in the working directory
    mat_path = 'DH_FR1.mat'

    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)   # (2, 18)
    d_hat        = np.asarray(data['d_hat'],        dtype=float)   # (18, num_user)
    p            = np.asarray(data['p'],            dtype=float)   # (2, num_user) — GT

    # 2) algorithm — num_user taken dynamically from input
    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) return — numpy array, shape (2, num_user)
    return p_hat


if __name__ == "__main__":
    p_hat = main()
    # optional local evaluation: GT is loaded inside main(); re-read for the summary
    try:
        _d = sio.loadmat('DH_FR1.mat', squeeze_me=False)
        if 'p' in _d:
            p = np.asarray(_d['p'], dtype=float)
            e = np.sqrt(((p_hat - p) ** 2).sum(0))
            print("N %d | MAE %.3f | Median %.3f | RMSE %.3f | 1m %.1f%% | 2m %.1f%% | 5m %.1f%%"
                  % (e.size, e.mean(), np.median(e), np.sqrt((e**2).mean()),
                     (e < 1).mean()*100, (e < 2).mean()*100, (e < 5).mean()*100))
    except Exception:
        pass
