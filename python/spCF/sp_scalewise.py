"""Scale-wise spatial process extraction."""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np


def sp_scalewise(mod, bw_range: Tuple[float, float] = (0.0, np.inf),
                 time_range: Tuple[float, float] = (-np.inf, np.inf)):
    """Extract the (multiscale) spatial process for bandwidths in ``bw_range``.

    Parameters
    ----------
    mod : CFLM | CFGLM | CFDGLM
        Output of :func:`spCF.cf_lm`, :func:`spCF.cf_glm` or
        :func:`spCF.cf_dglm`.
    bw_range : tuple (low, high)
        Half-open bandwidth interval ``[low, high)``: scales with bandwidth
        ``b`` such that ``low <= b < high`` are synthesized. The half-open
        convention lets contiguous ranges partition the scales without
        double-counting a shared endpoint. Default ``(0, inf)`` = all scales.
    time_range : tuple (low, high)
        Only used for a :func:`spCF.cf_dglm` (spatio-temporal) fit, which
        carries a time index for every row of ``Z``. The process is averaged
        over time points in ``[low, high]`` at each location. Ignored (with a
        warning if set to a non-default value) for purely spatial fits.

    Returns
    -------
    dict
        ``{"pred": {...}, "pred0": {...} | None}``. For a ``cf_dglm`` fit each
        entry additionally carries ``px``/``py`` coordinates and ``n_time``
        (number of averaged time points).
    """
    bands = np.asarray(mod.bands) if mod.bands is not None else np.empty(0)
    # Use min/max so a reversed range is tolerated, matching R's
    # min(bw_range)/max(bw_range).
    lo, hi = float(min(bw_range)), float(max(bw_range))
    cols = np.where((bands >= lo) & (bands < hi))[0]
    if cols.size == 0:
        raise ValueError("no spatial process detected within the bandwidth range (bw_range)")

    # per-row time index (present only for cf_dglm fits); None otherwise
    time = mod.other.get("time")
    has_time = time is not None
    if not has_time and (time_range[0] != -np.inf or time_range[1] != np.inf):
        warnings.warn("time_range is ignored: 'mod' has no time dimension (not a cf_dglm fit)")

    tlo, thi = float(min(time_range)), float(max(time_range))

    def collapse(Z, Zsd, coords, tvec):
        Z = np.asarray(Z, dtype=float)
        Zsd = np.asarray(Zsd, dtype=float)
        fld = Z[:, cols].sum(axis=1)                 # multiscale field per row
        vr = np.sum(Zsd[:, cols] ** 2, axis=1)        # its variance (scales indep.)
        if tvec is None:
            return {"pred": fld, "pred_sd": np.sqrt(vr)}
        tvec = np.asarray(tvec)
        keep = (tvec >= tlo) & (tvec <= thi)
        if not np.any(keep):
            raise ValueError("no observations within the specified time_range")
        fld = fld[keep]
        vr = vr[keep]
        co = np.asarray(coords, dtype=float)[keep]
        # group by unique location (order of first appearance)
        keys = co[:, 0].astype(str).astype(object) + "\r" + co[:, 1].astype(str).astype(object)
        _, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
        order = np.argsort(first_idx)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        g = rank[inv]                                # 0-based group in first-seen order
        ng = order.size
        nt = np.bincount(g, minlength=ng).astype(float)          # # averaged time points
        mn = np.bincount(g, weights=fld, minlength=ng) / nt      # time-mean of the field
        vm = np.bincount(g, weights=vr, minlength=ng) / nt ** 2  # variance of the time-mean
        uc = co[first_idx[order]]
        return {"px": uc[:, 0], "py": uc[:, 1], "pred": mn,
                "pred_sd": np.sqrt(vm), "n_time": nt}

    coords = mod.other.get("coords")
    z_ms = collapse(mod.Z, mod.Z_sd, coords, time if has_time else None)

    z0_ms = None
    n0 = mod.other.get("n0")
    if (n0 is not None and not (isinstance(n0, float) and np.isnan(n0))
            and mod.Z0 is not None):
        coords0 = mod.other.get("coords0")
        time0 = mod.other.get("time0") if has_time else None
        z0_ms = collapse(mod.Z0, mod.Z0_sd, coords0, time0)

    return {"pred": z_ms, "pred0": z0_ms}
