"""Explicit kilohertz conduction-block test.

A train of intracellular test pulses is delivered at one end of the axon and
counted at the other, while a kilohertz field is applied through the surface
electrodes.  The block threshold is the lowest carrier amplitude at which test
action potentials stop arriving.
"""
import numpy as np
from . import mrg as M
from . import waveforms as W
from . import coupling as C

_CACHE = {}


def _res(depth, fiberD, n_nodes, dispersion, lead_path):
    key = (depth, fiberD, n_nodes, dispersion, lead_path)
    if key not in _CACHE:
        ax = M.MRGAxon(fiberD, n_nodes=n_nodes)
        lf = C.LeadFields(lead_path)
        _CACHE[key] = (ax, lf.lead_matrix(ax.zc, depth, dispersion))
    return _CACHE[key]


def block_run(carrier, fb, amp_mA, depth=5.0, fiberD=10.0, n_nodes=81,
              dt_ms=0.0025, settle_ms=60.0, n_test=5, isi_ms=20.0,
              test_nA=5.0, test_pw_ms=0.1, dispersion="two_band",
              lead_path="leadfields.npz", return_traces=False):
    """carrier: 'blocker' (pure fb on both channels, antiphase -> cancels),
                'ch1_only'  (TSINS channel 1 alone: harmonics + blocker),
                'blocker_1ch' (fb on channel 1 only),
                'tis_1ch'   (fb on channel 1 only, same as blocker_1ch),
                'none'
    """
    ax, lead = _res(depth, fiberD, n_nodes, dispersion, lead_path)
    T = settle_ms + n_test * isi_ms + 20.0                  # ms
    nstep = int(round(T / dt_ms))
    t_ms = np.arange(nstep + 1) * dt_ms
    t_s = t_ms * 1e-3
    ivec = np.zeros((nstep + 1, 4))
    g, Ab = W.G_DEFAULT, W.KAPPA_DEFAULT
    env = np.clip((t_ms - 5.0) / 20.0, 0, 1)                # 20 ms ramp-on
    if carrier in ("blocker", "ch1_only", "blocker_1ch", "tis_1ch"):
        blk = g * Ab * amp_mA * np.sin(2 * np.pi * fb * t_s) * env
        if carrier == "blocker":
            ivec[:, 1] = blk
            ivec[:, 3] = -blk
        else:
            ivec[:, 1] = blk
        if carrier == "ch1_only":
            d = W.tsins_channels(t_s, fb)
            ivec[:, 0] = d["ch1_lf"] * amp_mA * env
    elif carrier != "none":
        raise ValueError(carrier)

    nodes = ax.node_index
    j_stim = int(nodes[3])
    j_det = int(nodes[-4])
    istim = np.zeros(nstep + 1)
    pulse_t = settle_ms + np.arange(n_test) * isi_ms
    for pt in pulse_t:
        istim[(t_ms >= pt) & (t_ms < pt + test_pw_ms)] = test_nA * 1e-3

    r = M.simulate(ax, lead, ivec, dt_ms, istim=istim, istim_idx=j_stim, rec_dt=0.05)
    v, tt = r["v"], r["t"]
    above = v > -20.0
    cross = above[1:] & ~above[:-1]
    j_s = int(np.where(nodes == j_stim)[0][0])
    j_d = int(np.where(nodes == j_det)[0][0])
    # count only APs after the first test pulse
    mask = tt[1:] > settle_ms - 1.0
    n_init = int(cross[mask, j_s].sum())
    n_arr = int(cross[mask, j_d].sum())
    # spontaneous firing during the settling period (carrier-driven onset response)
    n_onset = int(cross[~mask, j_d].sum())
    out = dict(carrier=carrier, fb=fb, amp_mA=amp_mA, depth=depth,
               n_initiated=n_init, n_arrived=n_arr, n_test=n_test,
               n_onset_aps=n_onset,
               blocked=bool(n_arr < n_test),
               transmission=n_arr / n_test,
               ve_peak_mV=float(np.abs(ivec @ lead.T).max()))
    if return_traces:
        out["traces"] = dict(t=tt, v=v, z=ax.node_positions(), ivec=ivec, t_ms=t_ms)
    return out


def block_threshold(carrier, fb, depth=5.0, lo=1.0, hi=200.0, n=14, **kw):
    """Lowest carrier amplitude (mA) at which fewer than all test APs arrive."""
    amps = np.geomspace(lo, hi, n)
    prev_ok = None
    for a in amps:
        r = block_run(carrier, fb, a, depth=depth, **kw)
        ok = not r["blocked"]
        if prev_ok is True and not ok:
            lo2, hi2 = a_prev, a
            for _ in range(6):
                mid = 0.5 * (lo2 + hi2)
                if block_run(carrier, fb, mid, depth=depth, **kw)["blocked"]:
                    hi2 = mid
                else:
                    lo2 = mid
            return float(hi2)
        prev_ok, a_prev = ok, a
    return np.nan
