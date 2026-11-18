"""Tissue dielectric properties (Gabriel 4-Cole-Cole) used by the TSINS model."""
import numpy as np

EPS0 = 8.8541878128e-12

# 4-Cole-Cole parameters, Gabriel, Lau & Gabriel (1996) / IT'IS database.
# (eps_inf, sigma_i, de1, tau1[ps], a1, de2, tau2[ns], a2, de3, tau3[us], a3, de4, tau4[ms], a4)
COLE_COLE = {
    "muscle":   (4.0, 0.2000,   50, 7.234, 0.10,  7000, 353.68, 0.10, 1.2e6, 318.31, 0.10, 2.5e7, 2.274,  0.00),
    "fat":      (2.5, 0.0100,    3, 7.958, 0.20,    15,  15.915, 0.10, 3.3e4, 159.16, 0.05, 1.0e7, 7.958,  0.01),
    "nerve":    (4.0, 0.0060,   26, 7.958, 0.10,   500, 106.10, 0.15, 7.0e4,  15.915, 0.20, 4.0e7, 15.915, 0.00),
    "skin_dry": (4.0, 0.0002,   32, 7.234, 0.00,  1100,  32.481, 0.20, 0.0,  159.16, 0.20, 0.0,  15.915, 0.20),
}


def cole_cole(tissue, f_hz):
    """Complex permittivity -> effective conductivity sigma_eff(f) in S/m."""
    e_inf, si, de1, t1, a1, de2, t2, a2, de3, t3, a3, de4, t4, a4 = COLE_COLE[tissue]
    taus = np.array([t1 * 1e-12, t2 * 1e-9, t3 * 1e-6, t4 * 1e-3])
    des = np.array([de1, de2, de3, de4])
    als = np.array([a1, a2, a3, a4])
    w = 2 * np.pi * np.asarray(f_hz, dtype=float)
    eps = e_inf + 0j
    for de, tau, a in zip(des, taus, als):
        eps = eps + de / (1.0 + (1j * w * tau) ** (1.0 - a))
    eps = eps + si / (1j * w * EPS0)
    return -np.imag(eps) * w * EPS0          # sigma_eff = -omega*eps0*Im(eps_complex)


# Defaults used by the model (8 kHz, as in the TSINS paper).
def default_sigmas(f_hz=8e3, wet_skin=True, sigma_skin_wet=0.10, sigma_gel=1.4):
    s = {
        "muscle": float(cole_cole("muscle", f_hz)),
        "fat": float(cole_cole("fat", f_hz)),
        "nerve": float(cole_cole("nerve", f_hz)),
        "gel": float(sigma_gel),
    }
    s["skin"] = float(sigma_skin_wet) if wet_skin else float(cole_cole("skin_dry", f_hz))
    s["skin_dry"] = float(cole_cole("skin_dry", f_hz))
    return s
