"""CAFA-style metrics: protein-centric Fmax, AUPR, and hierarchy-violation rate."""
import numpy as np
from sklearn.metrics import average_precision_score


def fmax(y_true, y_prob, thresholds=np.arange(0.01, 1.0, 0.01)):
    """Protein-centric Fmax. Precision averaged over proteins with >=1 prediction, recall over all proteins."""
    y_true = y_true.astype(bool)
    keep = y_true.sum(1) > 0
    y_true, y_prob = y_true[keep], y_prob[keep]
    n_true = y_true.sum(1)
    best, best_t = 0.0, 0.0
    for t in thresholds:
        pred = y_prob >= t
        n_pred = pred.sum(1)
        tp = (pred & y_true).sum(1)
        has = n_pred > 0
        if not has.any():
            continue
        prec = (tp[has] / n_pred[has]).mean()
        rec = (tp / n_true).mean()
        f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        if f > best:
            best, best_t = f, t
    return best, best_t


def aupr(y_true, y_prob):
    """micro AUPR (all pairs flattened) and macro AUPR (mean over terms having >=1 positive)."""
    micro = average_precision_score(y_true.ravel(), y_prob.ravel())
    cols = np.where(y_true.sum(0) > 0)[0]
    macro = float(np.mean([average_precision_score(y_true[:, j], y_prob[:, j]) for j in cols]))
    return micro, macro


def prf_at(y_true, y_prob, t):
    pred = y_prob >= t
    tp = (pred & y_true.astype(bool)).sum()
    p = tp / max(pred.sum(), 1)
    r = tp / max(y_true.sum(), 1)
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def violation_rate(y_prob, child_idx, parent_idx, t):
    """Fraction of predicted child terms (>=t) whose ancestor is NOT predicted -> biological inconsistency."""
    pred = y_prob >= t
    child_on = pred[:, child_idx]
    parent_off = ~pred[:, parent_idx]
    return float((child_on & parent_off).sum() / max(child_on.sum(), 1))
