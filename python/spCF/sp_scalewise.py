"""Scale-wise spatial process extraction."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def sp_scalewise(mod, bw_range: Tuple[float, float] = (0.0, np.inf)):
    """Extract spatial process for bandwidths in `bw_range` from a fitted CF model.

    Parameters
    ----------
    mod : CFLM | CFGLM
        Output of :func:`spCF.cf_lm` or :func:`spCF.cf_glm`.
    bw_range : tuple (low, high)
        Inclusive bandwidth range. Default ``(0, inf)`` synthesizes all scales.

    Returns
    -------
    dict
        ``{"pred": {"pred", "pred_sd"}, "pred0": {...} | None}``.
    """
    bands = np.asarray(mod.bands) if mod.bands is not None else np.empty(0)
    lo, hi = float(bw_range[0]), float(bw_range[1])
    cols = np.where((bands >= lo) & (bands <= hi))[0]
    if cols.size == 0:
        raise ValueError("no spatial process detected within bw_range")

    Z = np.asarray(mod.Z)
    Z_sd = np.asarray(mod.Z_sd)
    pred = Z[:, cols].sum(axis=1)
    pred_sd = np.sqrt(np.sum(Z_sd[:, cols] ** 2, axis=1))
    z_ms = {"pred": pred, "pred_sd": pred_sd}

    z0_ms = None
    n0 = mod.other.get("n0")
    if n0 is not None and mod.Z0 is not None:
        Z0 = np.asarray(mod.Z0)
        Z0_sd = np.asarray(mod.Z0_sd)
        pred0 = Z0[:, cols].sum(axis=1)
        pred0_sd = np.sqrt(np.sum(Z0_sd[:, cols] ** 2, axis=1))
        z0_ms = {"pred": pred0, "pred_sd": pred0_sd}

    return {"pred": z_ms, "pred0": z0_ms}
