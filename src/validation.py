"""
Modular validation / diagnostic utilities for DenseDetector models.

peak-level matching (TP / FN / FP), and provides tools to collect
false-negative ("missed peak") and false-positive ("spurious peak")
cases for hard-mining into a retraining or fine-tuning set.

CONFIRMED DATA FORMATS:

  1. `predict_peaks_dense(...)` returns a list of 4-tuples:
         (confidence, A, position, gamma)
     built from `candidates.append((presence[g], A, pos, gam))`.

  2. `peaks_test[b]` (from `generate_dataset`) is a list of 3-tuples:
         (A, position, gamma)

  Both use *position* (not grid index) in the same units as `cfg.W`.

  3. `cfg.w_grid` gives the frequency-shift value at the *start* of each
     grid cell. `position_to_cell` finds the nearest grid cell by
     nearest-value lookup on w_grid -- fine for diagnostics, but if you
     have an exact index formula elsewhere (e.g. from build_targets_batch),
     prefer that for consistency.
"""

import numpy as np
import torch

from configs import Config
from src.models_peak_predict import DenseDetector
from src.inference import predict_peaks_dense


# ---------------------------------------------------------------------------
#  Peak matching
# ---------------------------------------------------------------------------

def match_peaks(true_peaks, pred_peaks, tol):
    """
    Greedy nearest-neighbor matching between true and predicted peak
    positions (1D). Matches closest pairs first, subject to a distance
    tolerance, each peak used at most once.

    Parameters
    ----------
    true_peaks, pred_peaks : array-like of float positions
    tol : float
        Maximum distance to count as a match, in the same units as the
        positions (e.g. cm^-1).

    Returns
    -------
    matches : list of (true_idx, pred_idx)
    fn_idx  : list of true_peaks indices with no match (missed / FN)
    fp_idx  : list of pred_peaks indices with no match (spurious / FP)
    """
    true_peaks = np.asarray(true_peaks, dtype=float)
    pred_peaks = np.asarray(pred_peaks, dtype=float)

    if len(true_peaks) == 0 and len(pred_peaks) == 0:
        return [], [], []
    if len(true_peaks) == 0:
        return [], [], list(range(len(pred_peaks)))
    if len(pred_peaks) == 0:
        return [], list(range(len(true_peaks))), []

    dist = np.abs(true_peaks[:, None] - pred_peaks[None, :])

    matches = []
    used_true, used_pred = set(), set()

    # Greedy: consider pairs in ascending distance order, skip if either
    # side is already used, stop once remaining distances exceed tol.
    flat_order = np.argsort(dist, axis=None)
    for flat_idx in flat_order:
        i, j = np.unravel_index(flat_idx, dist.shape)
        if dist[i, j] > tol:
            break
        # i or j already assigned 
        if i in used_true or j in used_pred:
            continue
        matches.append((int(i), int(j)))
        used_true.add(i)
        used_pred.add(j)

    fn_idx = [i for i in range(len(true_peaks)) if i not in used_true]
    fp_idx = [j for j in range(len(pred_peaks)) if j not in used_pred]
    return matches, fn_idx, fp_idx


def position_to_cell(position, cfg):
    """Nearest grid-cell index for a peak position, via cfg.w_grid."""
    return int(np.argmin(np.abs(cfg.w_grid - position)))


# ---------------------------------------------------------------------------
#  Diagnostic evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, cfg, device, X_raw, peaks_list, tol=None):
    """
    Run peak-level diagnostic validation over a set of profiles.

    Parameters
    ----------
    X_raw : array-like or tensor, shape (n_samples, n_points)
        Raw input signals (single-channel convention, same as
        `X_test_raw` in main.py).
    peaks_list : list, len == n_samples
        True peak positions per profile (see module docstring assumption 2).
    tol : float, optional
        Matching tolerance. Defaults to half a grid-cell spacing.

    Returns
    -------
    dict with:
        'summary'    : precision / recall / F1 / counts
        'fn_records' : list of dicts, one per missed peak
        'fp_records' : list of dicts, one per spurious prediction
    """
    if tol is None:
        tol = 0.5 * abs(cfg.w_grid[1] - cfg.w_grid[0])

    n_samples = len(peaks_list)
    fn_records, fp_records = [], []
    total_tp = total_fn = total_fp = 0

    model.eval()
    for b in range(n_samples):
        x_raw = X_raw[b]
        x_np = x_raw.numpy() if torch.is_tensor(x_raw) else np.asarray(x_raw)

        # true peaks: list of (A, position, gamma)
        true_tuples = list(peaks_list[b])
        true_peaks = np.array([t[1] for t in true_tuples], dtype=float)

        # predicted candidates: list of (confidence, A, position, gamma)
        detected = predict_peaks_dense(model, x_np, cfg, device=device)
        detected = list(detected)
        pred_peaks = np.array([d[2] for d in detected], dtype=float)

        matches, fn_idx, fp_idx = match_peaks(true_peaks, pred_peaks, tol)
        total_tp += len(matches)
        total_fn += len(fn_idx)
        total_fp += len(fp_idx)

        if not fn_idx and not fp_idx:
            continue  # clean profile, nothing to record

        # Only pull raw model outputs for profiles that actually had an error
        with torch.no_grad():
            x_t = torch.as_tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
            presence, offset, amp, gamma = model(x_t)
        presence_np = presence.squeeze(0).cpu().numpy()

        true_cells = [position_to_cell(p, cfg) for p in true_peaks]

        for i in fn_idx:
            A_true, pos_true, gam_true = true_tuples[i]
            cell = true_cells[i]
            collision = true_cells.count(cell) > 1
            fn_records.append({
                "sample_idx": b,
                "position": float(pos_true),
                "amplitude": float(A_true),
                "gamma": float(gam_true),
                "grid_cell": cell,
                "raw_presence": float(presence_np[cell]),
                "grid_collision": collision,
                "n_true_in_profile": len(true_peaks),
                "n_pred_in_profile": len(pred_peaks),
                "X_raw": x_np,
                "true_peaks": true_tuples,
            })

        for j in fp_idx:
            conf_pred, A_pred, pos_pred, gam_pred = detected[j]
            cell = position_to_cell(pos_pred, cfg)
            fp_records.append({
                "sample_idx": b,
                "position": float(pos_pred),
                "amplitude": float(A_pred),
                "gamma": float(gam_pred),
                "confidence": float(conf_pred),
                "grid_cell": cell,
                "raw_presence": float(presence_np[cell]),
                "n_true_in_profile": len(true_peaks),
                "n_pred_in_profile": len(pred_peaks),
                "X_raw": x_np,
                "true_peaks": true_tuples,
            })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else float("nan")
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and not np.isnan(precision + recall) else float("nan"))

    summary = {
        "n_samples": n_samples,
        "tp": total_tp, "fn": total_fn, "fp": total_fp,
        "precision": precision, "recall": recall, "f1": f1,
    }
    return {"summary": summary, "fn_records": fn_records, "fp_records": fp_records}


# ---------------------------------------------------------------------------
#  Hard-mining: build a retraining/fine-tuning set from flagged profiles
# ---------------------------------------------------------------------------

def build_hard_mined_set(fn_records, fp_records=None, include_fp_profiles=True):
    """
    Collect unique profiles that produced at least one FN (and optionally
    FP) into a dataset ready for build_targets_batch + retraining/fine-tuning.

    Dedupes by sample_idx so a profile with multiple missed peaks is only
    included once. `true_peaks` in each record is already the original
    list of (A, position, gamma) tuples for that profile -- the same
    format `generate_dataset` / `build_targets_batch` expect -- so
    `peaks_hard` can be fed straight into `build_targets_batch` alongside
    X_hard, no reconstruction needed.

    Returns
    -------
    X_hard : np.ndarray, shape (n_hard, n_points)
    peaks_hard : list, one (A, position, gamma) tuple-list per profile
    """
    seen = {}
    for r in fn_records:
        seen[r["sample_idx"]] = (r["X_raw"], r["true_peaks"])
    if include_fp_profiles and fp_records:
        for r in fp_records:
            seen.setdefault(r["sample_idx"], (r["X_raw"], r["true_peaks"]))

    X_hard = np.stack([v[0] for v in seen.values()])
    peaks_hard = [v[1] for v in seen.values()]
    return X_hard, peaks_hard
