"""
Semi-analytic reference solution for the layered cylinder.

Solves div(sigma grad phi) = 0 in a radially layered cylinder of length L with
insulating end faces, driven by a prescribed normal current density on the
curved surface.  Expanding in a Fourier series around theta and a cosine series
along z (which satisfies the insulating end condition exactly) reduces the
problem to a modified-Bessel two-point boundary problem in r for every mode.

This carries no mesh error, so it is the reference against which the
finite-volume solver is verified.  It cannot represent the nerve inclusion or
the gel pads, which the finite-volume model adds.
"""
import numpy as np
from scipy.fft import fft, ifft, dct, idct
from scipy.special import ive, kve


def _Iratio(m, x, xref):
    """I_m(x) / I_m(xref) for x <= xref, overflow-free."""
    return ive(m, x) / ive(m, xref) * np.exp(x - xref)


def _dIratio(m, x, xref):
    """I'_m(x) / I_m(xref)."""
    return 0.5 * (ive(m - 1, x) + ive(m + 1, x)) / ive(m, xref) * np.exp(x - xref)


def _Kratio(m, x, xref):
    """K_m(x) / K_m(xref) for x >= xref, overflow-free."""
    return kve(m, x) / kve(m, xref) * np.exp(xref - x)


def _dKratio(m, x, xref):
    """K'_m(x) / K_m(xref)."""
    return -0.5 * (kve(m - 1, x) + kve(m + 1, x)) / kve(m, xref) * np.exp(xref - x)


class LayeredCylinder:
    """radii: outer radius of each layer (ascending); sigmas: one per layer."""

    def __init__(self, radii, sigmas, length, ntheta=128, nz=256):
        self.radii = np.asarray(radii, float)      # mm
        self.sig = np.asarray(sigmas, float)       # S/m
        self.L = float(length)                     # mm
        self.nt, self.nz = ntheta, nz
        self.theta = (np.arange(ntheta) + 0.5) * 2 * np.pi / ntheta - np.pi
        self.z = -self.L / 2 + (np.arange(nz) + 0.5) * self.L / nz
        self.m = np.fft.fftfreq(ntheta, d=1.0 / ntheta).astype(int)
        self.k = np.arange(nz) * np.pi / self.L   # 1/mm

    # ------------------------------------------------------------------
    def _mode_profile(self, m, k, r_eval):
        """phi(r_eval) for unit sigma*dphi/dr at the surface, one (m,k) mode."""
        R = self.radii
        nl = len(R)
        if k == 0 and m == 0:
            return None                                  # handled separately
        # state = (phi, sigma dphi/dr) propagated outward, starting A1 = 1
        if k > 0:
            xref = k * R[0]
            phi = _Iratio(abs(m), k * np.asarray([R[0]]), xref)[0]
            dph = self.sig[0] * k * _dIratio(abs(m), k * np.asarray([R[0]]), xref)[0]
            coeffs = [("I", 1.0, 0.0, k * R[0])]
        else:                                            # k = 0, m != 0
            mm = abs(m)
            phi = R[0] ** mm
            dph = self.sig[0] * mm * R[0] ** (mm - 1)
            coeffs = [("P", 1.0, 0.0, None)]
        for j in range(1, nl):
            ra, rb = R[j - 1], R[j]
            if k > 0:
                mm = abs(m)
                u = _Iratio(mm, k * np.asarray([ra, rb]), k * rb)
                du = k * _dIratio(mm, k * np.asarray([ra, rb]), k * rb)
                v = _Kratio(mm, k * np.asarray([ra, rb]), k * ra)
                dv = k * _dKratio(mm, k * np.asarray([ra, rb]), k * ra)
            else:
                mm = abs(m)
                u = np.array([ra, rb]) ** mm / rb ** mm
                du = mm * np.array([ra, rb]) ** (mm - 1) / rb ** mm
                v = np.array([ra, rb]) ** (-mm) / ra ** (-mm)
                dv = -mm * np.array([ra, rb]) ** (-mm - 1) / ra ** (-mm)
            M = np.array([[u[0], v[0]], [self.sig[j] * du[0], self.sig[j] * dv[0]]])
            A, B = np.linalg.solve(M, np.array([phi, dph]))
            phi = A * u[1] + B * v[1]
            dph = self.sig[j] * (A * du[1] + B * dv[1])
            coeffs.append(("IK", A, B, (ra, rb)))
        scale = 1.0 / dph                               # normalise to unit flux
        # evaluate at r_eval
        out = np.zeros_like(np.atleast_1d(r_eval), dtype=float)
        rq = np.atleast_1d(np.asarray(r_eval, float))
        for j in range(nl):
            lo = 0.0 if j == 0 else R[j - 1]
            sel = (rq > lo - 1e-12) & (rq <= R[j] + 1e-12)
            if not sel.any():
                continue
            rr = rq[sel]
            if j == 0:
                if k > 0:
                    out[sel] = _Iratio(abs(m), k * rr, k * R[0])
                else:
                    out[sel] = rr ** abs(m) / R[0] ** abs(m)
            else:
                _, A, B, (ra, rb) = coeffs[j]
                if k > 0:
                    out[sel] = A * _Iratio(abs(m), k * rr, k * rb) + \
                               B * _Kratio(abs(m), k * rr, k * ra)
                else:
                    mm = abs(m)
                    out[sel] = A * rr ** mm / rb ** mm + B * rr ** (-mm) / ra ** (-mm)
        return out * scale

    # ------------------------------------------------------------------
    def solve_line(self, J_surface, r_eval, theta_eval):
        """phi(z) on the line (r_eval, theta_eval) for surface density J(theta, z).

        J_surface: (ntheta, nz) array of inward current density in A/mm^2.
        Returns phi in volts on self.z (mm grid), with the mean removed.
        """
        Jh = fft(J_surface, axis=0)                      # theta modes
        Jh = dct(Jh, type=2, axis=1, norm=None) / (2 * self.nz)   # z cosine modes
        Jh[:, 1:] *= 2.0                                 # DCT-II -> cosine amplitudes
        phi_h = np.zeros_like(Jh)
        cache = {}
        for im, m in enumerate(self.m):
            for iq in range(self.nz):
                if abs(Jh[im, iq]) < 1e-18:
                    continue
                k = self.k[iq]
                if m == 0 and iq == 0:
                    continue                             # nullspace: zero mean
                key = (abs(m), iq)
                if key not in cache:
                    cache[key] = self._mode_profile(abs(m), k, [r_eval])[0]
                phi_h[im, iq] = Jh[im, iq] * cache[key]
        # synthesise at (r_eval, theta_eval)
        dth = 2 * np.pi / self.nt
        phase = np.exp(1j * self.m * (theta_eval + np.pi - dth / 2)) / self.nt
        cos_basis = np.cos(self.k[None, :] * (self.z[:, None] + self.L / 2))
        amp = (phi_h * phase[:, None]).sum(axis=0)       # (nz,) complex
        phi = cos_basis @ amp.real
        # J is given in A/mm^2 over areas in mm^2; sigma in S/m -> volts need 1e3
        return (phi - phi.mean()) * 1e3
