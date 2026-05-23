"""Conformal prediction utilities (split conformal + CQR)."""
import json
from pathlib import Path
from typing import Tuple

import numpy as np

_DEFAULT_THRESHOLDS = Path(__file__).parent.parent / "checkpoints" / "conformal_thresholds.json"


def load_thresholds(path: str = None) -> dict:
    p = Path(path) if path else _DEFAULT_THRESHOLDS
    with open(p) as f:
        return json.load(f)


def split_conformal_interval(
    point_pred: float,
    q_hat: float,
) -> Tuple[float, float]:
    """Symmetric split conformal interval: [pred - q_hat, pred + q_hat]."""
    return max(0.0, point_pred - q_hat), point_pred + q_hat


def cqr_interval(
    q10: float,
    q90: float,
    q_hat_cqr: float,
) -> Tuple[float, float]:
    """CQR interval: [Q10 - q_hat_cqr, Q90 + q_hat_cqr]."""
    return max(0.0, q10 - q_hat_cqr), q90 + q_hat_cqr


def conformal_threshold(cal_scores: np.ndarray, alpha: float = 0.10) -> float:
    """Compute finite-sample corrected conformal threshold from calibration scores."""
    n = len(cal_scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(cal_scores, level))
