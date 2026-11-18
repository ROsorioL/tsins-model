"""Small numerical helpers (scipy.interpolate/optimize are unavailable here)."""
import numpy as np
from scipy.linalg import solve_banded


def natural_cubic_coeffs(x, y):
    """Natural cubic spline second derivatives at the knots."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    h = np.diff(x)
    ab = np.zeros((3, n)); rhs = np.zeros(n)
    ab[1, 0] = 1.0; ab[1, -1] = 1.0                      # natural ends
    ab[1, 1:-1] = (h[:-1] + h[1:]) / 3.0
    ab[0, 2:] = h[1:] / 6.0                              # super-diagonal
    ab[2, :-2] = h[:-1] / 6.0                            # sub-diagonal
    rhs[1:-1] = (y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1]
    return solve_banded((1, 1), ab, rhs)


class CubicSpline1D:
    """Natural cubic spline; linear extrapolation outside the knot range."""

    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.m = natural_cubic_coeffs(self.x, self.y)
        self.h = np.diff(self.x)

    def __call__(self, xq):
        x, y, m, h = self.x, self.y, self.m, self.h
        xq = np.asarray(xq, float)
        i = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
        dx = xq - x[i]
        hi = h[i]
        a = (y[i + 1] - y[i]) / hi - hi * (2 * m[i] + m[i + 1]) / 6.0
        return y[i] + dx * (a + dx * (m[i] / 2.0 + dx * (m[i + 1] - m[i]) / (6.0 * hi)))

    def derivative(self, xq, order=1):
        x, y, m, h = self.x, self.y, self.m, self.h
        xq = np.asarray(xq, float)
        i = np.clip(np.searchsorted(x, xq) - 1, 0, x.size - 2)
        dx = xq - x[i]; hi = h[i]
        if order == 2:
            return m[i] + dx * (m[i + 1] - m[i]) / hi
        a = (y[i + 1] - y[i]) / hi - hi * (2 * m[i] + m[i + 1]) / 6.0
        return a + dx * (m[i] + dx * (m[i + 1] - m[i]) / (2.0 * hi))


def trilinear(grid_axes, values, pts):
    """Trilinear interpolation on a rectilinear grid; nearest-edge clamping."""
    out = np.zeros(len(pts))
    axes = [np.asarray(a, float) for a in grid_axes]
    idx, frac = [], []
    for d, a in enumerate(axes):
        q = np.asarray(pts)[:, d]
        i = np.clip(np.searchsorted(a, q) - 1, 0, a.size - 2)
        t = np.clip((q - a[i]) / (a[i + 1] - a[i]), 0.0, 1.0)
        idx.append(i); frac.append(t)
    for c in range(8):
        bx, by, bz = (c >> 2) & 1, (c >> 1) & 1, c & 1
        w = ((1 - frac[0]) if bx == 0 else frac[0]) * \
            ((1 - frac[1]) if by == 0 else frac[1]) * \
            ((1 - frac[2]) if bz == 0 else frac[2])
        out += w * values[idx[0] + bx, idx[1] + by, idx[2] + bz]
    return out


def find_threshold(test, lo, hi, n_coarse=12, tol_rel=0.02, max_iter=30):
    """Lowest amplitude in [lo, hi] at which `test` is True.

    A geometric ascending scan locates the first firing amplitude, then a
    bisection refines the bracket.  The scan is used instead of a plain
    bisection because kHz stimulation is not monotone in amplitude: an axon
    that fires at threshold can be conduction-blocked at higher amplitudes.

    Returns (threshold, scan_amplitudes, scan_results).  threshold is nan when
    nothing in [lo, hi] fires.
    """
    amps = np.geomspace(lo, hi, n_coarse)
    flags = []
    first = -1
    for i, a in enumerate(amps):
        f = bool(test(a))
        flags.append(f)
        if f and first < 0:
            first = i
            break
    flags = np.array(flags + [False] * (len(amps) - len(flags)))
    if first < 0:
        return np.nan, amps, flags
    if first == 0:
        return float(amps[0]), amps, flags
    a_lo, a_hi = amps[first - 1], amps[first]
    it = 0
    while (a_hi - a_lo) / a_hi > tol_rel and it < max_iter:
        mid = 0.5 * (a_lo + a_hi)
        if test(mid):
            a_hi = mid
        else:
            a_lo = mid
        it += 1
    return float(a_hi), amps, flags


def upper_block_threshold(test, lo, hi, n=16):
    """Lowest amplitude above the firing threshold at which the axon stops firing.

    Scans upward from `lo`; returns nan if firing persists to `hi`.
    """
    amps = np.geomspace(lo, hi, n)
    prev = None
    for a in amps:
        f = bool(test(a))
        if prev is True and f is False:
            return float(a)
        prev = f
    return np.nan
