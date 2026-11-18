"""Couple the finite-element lead fields to the MRG axon."""
import numpy as np
from .numerics import CubicSpline1D

LEAD_KEYS = ["ch1_lf", "ch1_hf", "ch2_lf", "ch2_hf"]


class LeadFields:
    """phi(z) along the nerve per 1 mA of channel current, in mV/mA."""

    def __init__(self, path="leadfields.npz"):
        d = np.load(path)
        self.z = d["z"].astype(float)              # mm
        self.depths = d["depths"]
        self._d = {k: d[k] for k in d.files}

    def available_depths(self):
        return list(self.depths)

    def phi(self, band, ch, depth):
        key = f"{band}_{ch}_d{int(round(depth))}"
        if key not in self._d:
            raise KeyError(f"{key} not in lead-field file; have depths {self.depths}")
        return self._d[key].astype(float) * 1e3    # V/mA -> mV/mA

    def lead_matrix(self, z_comp, depth, dispersion="two_band"):
        """(n_comp, 4) lead field in mV per mA for [ch1_lf, ch1_hf, ch2_lf, ch2_hf]."""
        cols = []
        for key in LEAD_KEYS:
            ch, band = key.split("_")
            b = band if dispersion == "two_band" else "hf"
            cols.append(CubicSpline1D(self.z, self.phi(b, ch, depth))(z_comp))
        return np.column_stack(cols)

    def channel_weight_ratio(self, depth, band="hf"):
        """Peak |phi| of ch1 relative to ch2 at the nerve."""
        a = np.abs(self.phi(band, "ch1", depth)).max()
        b = np.abs(self.phi(band, "ch2", depth)).max()
        return a / b


def extracellular_potential(lead, ivec):
    """Ve(t, compartment) in mV. lead (n_comp,4) mV/mA, ivec (nt,4) mA."""
    return ivec @ lead.T


def residual_at(f_hz, t, ve):
    """Amplitude of the spectral component of ve(t) at f_hz (mV)."""
    n = ve.size
    w = np.exp(-2j * np.pi * f_hz * t)
    return 2.0 * np.abs((ve * w).sum()) / n


def reconstruction_metrics(t, ve_sum, target, fb, f0=2.0, n0=4, K=7):
    """Fidelity of the in-tissue reconstruction against the ideal harmonic sum."""
    ve = ve_sum - ve_sum.mean()
    tg = target - target.mean()
    scale = (ve @ tg) / (tg @ tg) if tg @ tg > 0 else 0.0
    resid = ve - scale * tg
    nrmse = np.sqrt((resid ** 2).mean()) / (np.abs(ve).max() + 1e-30)
    harm = np.sqrt(sum(residual_at(n * f0, t, ve) ** 2 for n in range(n0, n0 + 2 * K)))
    blk = residual_at(fb, t, ve)
    return dict(nrmse=float(nrmse), harmonic_rms_mV=float(harm),
                blocking_mV=float(blk),
                blocking_ratio=float(blk / (harm + 1e-30)),
                peak_mV=float(np.abs(ve).max()))
