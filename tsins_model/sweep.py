"""Parameter sweeps over the TSINS / TIS / biphasic conditions."""
import numpy as np
from . import mrg as M
from . import waveforms as W
from . import coupling as C
from .numerics import find_threshold

_CACHE = {}
WINDOW = (0.25, 0.75)        # s; one 2 Hz period, every waveform starts at zero
PERIOD = WINDOW[1] - WINDOW[0]


def _resources(depth, fiberD, n_nodes, dispersion, lead_path):
    key = (depth, fiberD, n_nodes, dispersion, lead_path)
    if key not in _CACHE:
        ax = M.MRGAxon(fiberD, n_nodes=n_nodes)
        lf = C.LeadFields(lead_path)
        _CACHE[key] = (ax, lf.lead_matrix(ax.zc, depth, dispersion))
    return _CACHE[key]


def run_one(waveform, fb, amp_mA, depth=5.0, fiberD=10.0, n_nodes=51,
            dt_ms=0.0025, dispersion="two_band", rate=2.0, lead_path="leadfields.npz",
            window=WINDOW, rec_dt=0.05, return_traces=False):
    """One stimulation run; returns AP counts, propagation and timing."""
    ax, lead = _resources(depth, fiberD, n_nodes, dispersion, lead_path)
    t = np.arange(window[0], window[1] + 1e-12, dt_ms * 1e-3)
    ivec, _, d = W.make_drive(waveform, t, fb, amp_mA, rate=rate)
    r = M.simulate(ax, lead, ivec, dt_ms, rec_dt=rec_dt)
    v, tt = r["v"], r["t"]
    above = v > -20.0
    cross = above[1:] & ~above[:-1]
    counts = cross.sum(axis=0)
    n_end = int(min(counts[2], counts[-3]))
    first = np.full(counts.size, np.nan)
    idx = np.argmax(cross, axis=0)
    hit = counts > 0
    first[hit] = tt[idx[hit]]
    out = dict(waveform=waveform, fb=fb, amp_mA=amp_mA, depth=depth, fiberD=fiberD,
               n_ap_max=int(counts.max()), n_ap_propagated=n_end,
               frac_nodes_firing=float((counts > 0).mean()),
               ap_rate_hz=n_end / PERIOD,
               ve_peak_mV=float(np.abs(ivec @ lead.T).max()),
               charge_mC=float(W.delivered_charge(t, ivec)))
    if hit.any():
        j0 = int(np.nanargmin(first))
        out["init_z_mm"] = float(ax.zc[ax.node_index[j0]])
        out["first_ap_ms"] = float(np.nanmin(first))
    else:
        out["init_z_mm"] = np.nan
        out["first_ap_ms"] = np.nan
    if return_traces:
        out["traces"] = dict(t=tt, v=v, z=ax.node_positions(),
                             ve=(ivec @ lead.T), t_drive=t, ivec=ivec, target=d["target"])
    return out


def threshold(waveform, fb, depth=5.0, lo=0.2, hi=40.0, n_coarse=14,
              criterion="propagated", **kw):
    """Lowest channel amplitude (mA) producing a propagated action potential."""
    def test(a):
        r = run_one(waveform, fb, a, depth=depth, **kw)
        return (r["n_ap_propagated"] > 0 if criterion == "propagated"
                else r["n_ap_max"] > 0)
    th, amps, flags = find_threshold(test, lo, hi, n_coarse=n_coarse, tol_rel=0.03)
    return th


def _job(a):
    kind = a.pop("_kind")
    return (kind, run_one(**a)) if kind == "run" else (kind, dict(a, value=threshold(**a)))
