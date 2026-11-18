"""
Three-layer cylindrical forearm volume conductor for the TSINS / TIS model.

Geometry (after Osorio et al., extended from the paper's single muscle domain):
    muscle core      r < 20.5 mm
    subcutaneous fat 20.5 - 23.5 mm
    skin             23.5 - 25.0 mm
    electrode gel    25.0 - 25.5 mm, present only under the contact footprints
    nerve            cylinder of radius 1.5 mm parallel to the axis, at depth d
                     below the skin surface (d = 25 mm puts it on the axis, as
                     in the published model)

Quasi-static conduction:  div(sigma grad phi) = 0, insulating boundaries
everywhere except the gel top faces, where the terminal current is injected.
Discretised with a cell-centred finite-volume scheme on a graded (r, theta, z)
grid and solved with an algebraic-multigrid-preconditioned CG.
"""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from .numerics import trilinear, CubicSpline1D


# ----------------------------------------------------------------- geometry --
@dataclass
class Geometry:
    r_muscle: float = 20.5     # mm, muscle/fat interface
    r_fat: float = 23.5        # mm, fat/skin interface
    r_skin: float = 25.0       # mm, skin surface (= Rarm in the paper)
    t_gel: float = 0.5         # mm, gel thickness under each contact
    length: float = 200.0      # mm, cylinder length (L_arm in the paper)
    r_nerve: float = 1.5       # mm, nerve radius
    nerve_depth: float = 5.0   # mm, depth of the nerve axis below the skin
    pin_radius: float = 4.0    # mm, radius of one contact pin footprint (~0.5 cm^2)
    pin_pair_sep: float = 10.0 # mm, arc separation of the two paired pins
    contact_z: float = 10.0    # mm, axial offset of each contact from z = 0
    contact_s: float = 10.0    # mm, circumferential (arc) offset of each contact
    matrix_offset_s: float = 0.0  # mm, shift of the whole matrix off the nerve

    @property
    def r_nerve_axis(self):
        """Radial coordinate of the nerve axis (mm)."""
        return self.r_skin - self.nerve_depth


def contact_centres(g: Geometry):
    """(z, arc) centres of the four contacts; cross-paired diagonal layout.

    Channel 1 runs (-z, -s) -> (+z, +s); channel 2 runs (-z, +s) -> (+z, -s).
    The nerve lies at arc = 0, midway between the two diagonals.
    """
    z, s, o = g.contact_z, g.contact_s, g.matrix_offset_s
    return {
        ("ch1", +1): (-z, -s + o), ("ch1", -1): (+z, +s + o),
        ("ch2", +1): (-z, +s + o), ("ch2", -1): (+z, -s + o),
    }


def pin_centres(g: Geometry):
    """Each contact is two adjacent pins wired together (arc separation d)."""
    out = {}
    for key, (zc, sc) in contact_centres(g).items():
        out[key] = [(zc, sc - g.pin_pair_sep / 2), (zc, sc + g.pin_pair_sep / 2)]
    return out


# --------------------------------------------------------------------- grid --
def _graded(a, b, n_fine, x_fine, growth):
    """Faces from a to b: uniform spacing up to x_fine, then geometric growth."""
    faces = list(np.linspace(a, x_fine, n_fine + 1))
    h = faces[1] - faces[0]
    x = x_fine
    while x < b - 1e-9:
        h = h * growth
        x = min(b, x + h)
        faces.append(x)
    return np.array(faces)


def make_grid(g: Geometry, nr_muscle=28, nr_fat=5, nr_skin=4, nr_gel=2,
              ntheta=96, dz_fine=1.0, z_fine=30.0, z_growth=1.18):
    r_f = np.concatenate([
        np.linspace(0.0, g.r_muscle, nr_muscle + 1),
        np.linspace(g.r_muscle, g.r_fat, nr_fat + 1)[1:],
        np.linspace(g.r_fat, g.r_skin, nr_skin + 1)[1:],
        np.linspace(g.r_skin, g.r_skin + g.t_gel, nr_gel + 1)[1:],
    ])
    th_f = np.linspace(-np.pi, np.pi, ntheta + 1)
    half = _graded(0.0, g.length / 2, int(round(z_fine / dz_fine)), z_fine, z_growth)
    z_f = np.concatenate([-half[::-1], half[1:]])
    return r_f, th_f, z_f


# ------------------------------------------------------------------- model --
class VolumeConductor:
    """Assembles and solves the quasi-static problem on a (r, theta, z) grid."""

    def __init__(self, geom: Geometry, sigmas: dict, grid=None,
                 include_nerve=True, include_gel=True, verbose=False):
        self.g = geom
        self.sig = sigmas
        self.include_nerve = include_nerve
        self.include_gel = include_gel
        self.r_f, self.th_f, self.z_f = grid if grid is not None else make_grid(geom)
        self.verbose = verbose
        self._build()

    # -- grid metrics ------------------------------------------------------
    def _build(self):
        g = self.g
        r_f, th_f, z_f = self.r_f, self.th_f, self.z_f
        self.nr, self.nt, self.nz = len(r_f) - 1, len(th_f) - 1, len(z_f) - 1
        self.r_c = 0.5 * (r_f[1:] + r_f[:-1])
        self.th_c = 0.5 * (th_f[1:] + th_f[:-1])
        self.z_c = 0.5 * (z_f[1:] + z_f[:-1])
        self.dr = np.diff(r_f)
        self.dth = th_f[1] - th_f[0]
        self.dz = np.diff(z_f)

        R, T, Z = np.meshgrid(self.r_c, self.th_c, self.z_c, indexing="ij")

        # --- conductivity field and active mask ---------------------------
        sigma = np.full((self.nr, self.nt, self.nz), self.sig["muscle"])
        sigma[(R > g.r_muscle) & (R <= g.r_fat)] = self.sig["fat"]
        sigma[(R > g.r_fat) & (R <= g.r_skin)] = self.sig["skin"]
        gel = R > g.r_skin
        sigma[gel] = self.sig["gel"]

        # nerve: cylinder parallel to the axis at (r_n, theta = 0)
        if self.include_nerve:
            xn, yn = g.r_nerve_axis, 0.0
            X, Y = R * np.cos(T), R * np.sin(T)
            inside = ((X - xn) ** 2 + (Y - yn) ** 2) <= g.r_nerve ** 2
            sigma[inside] = self.sig["nerve"]
            self.nerve_mask = inside
        else:
            self.nerve_mask = np.zeros_like(sigma, dtype=bool)

        # gel exists only under the contact pins
        pin_mask = np.zeros((self.nt, self.nz), dtype=bool)
        self.pin_masks = {}
        arc = self.th_c * g.r_skin                       # arc coordinate at the skin
        A_arc, A_z = np.meshgrid(arc, self.z_c, indexing="ij")
        for key, pins in pin_centres(g).items():
            m = np.zeros((self.nt, self.nz), dtype=bool)
            for (zc, sc) in pins:
                m |= ((A_arc - sc) ** 2 + (A_z - zc) ** 2) <= g.pin_radius ** 2
            self.pin_masks[key] = m
            pin_mask |= m
        active = np.ones_like(sigma, dtype=bool)
        gel_rings = np.where(gel[:, 0, 0])[0]         # radial shells that are gel
        if self.include_gel and gel_rings.size:
            active[gel_rings, :, :] = pin_mask[None, :, :]
            self.term_ring = int(gel_rings[-1])
        else:
            if gel_rings.size:
                active[gel_rings, :, :] = False
            self.term_ring = int(np.where(self.r_c <= g.r_skin)[0][-1])
        self.sigma, self.active = sigma, active
        self.n_active = int(active.sum())

        idx = -np.ones(active.shape, dtype=np.int64)
        idx[active] = np.arange(self.n_active)
        self.idx = idx

        self.vol = (0.5 * (r_f[1:] ** 2 - r_f[:-1] ** 2))[:, None, None] * self.dth * self.dz[None, None, :]
        self._assemble()

    # -- matrix ------------------------------------------------------------
    def _assemble(self):
        nr, nt, nz = self.nr, self.nt, self.nz
        r_f, dr, dth, dz = self.r_f, self.dr, self.dth, self.dz
        r_c, sig, idx = self.r_c, self.sigma, self.idx
        rows, cols, vals = [], [], []

        def add(i1, i2, G):
            ok = (i1 >= 0) & (i2 >= 0) & (G > 0)
            a, b, g_ = i1[ok], i2[ok], G[ok]
            rows.extend([a, b, a, b]); cols.extend([a, b, b, a])
            vals.extend([g_, g_, -g_, -g_])

        # radial faces
        A = (r_f[1:-1])[:, None, None] * dth * dz[None, None, :] * np.ones((1, nt, 1))
        d1 = (r_f[1:-1] - r_c[:-1])[:, None, None]
        d2 = (r_c[1:] - r_f[1:-1])[:, None, None]
        G = A / (d1 / sig[:-1] + d2 / sig[1:])
        add(idx[:-1].ravel(), idx[1:].ravel(), G.ravel())

        # theta faces (periodic)
        A = (dr[:, None, None] * dz[None, None, :]) * np.ones((1, nt, 1))
        d = (r_c[:, None, None] * dth / 2) * np.ones((1, nt, 1))
        s1 = sig
        s2 = np.roll(sig, -1, axis=1)
        G = A / (d / s1 + d / s2)
        add(idx.ravel(), np.roll(idx, -1, axis=1).ravel(), G.ravel())

        # axial faces
        A = (0.5 * (r_f[1:] ** 2 - r_f[:-1] ** 2))[:, None, None] * dth * np.ones((1, nt, 1))
        d1 = (dz[:-1] / 2)[None, None, :]
        d2 = (dz[1:] / 2)[None, None, :]
        G = A / (d1 / sig[:, :, :-1] + d2 / sig[:, :, 1:])
        add(idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel(), G.ravel())

        n = self.n_active
        self.A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n)).tocsr()
        self.A.sum_duplicates()
        # mm -> m: conductances above are in S*mm; multiply by 1e-3 to get S
        self.A = self.A * 1e-3

    # -- right-hand side ---------------------------------------------------
    def rhs(self, channel, current_A=1e-3):
        """Injected-current vector for one channel (A). Anode +I, cathode -I."""
        b = np.zeros(self.n_active)
        top = self.term_ring
        dA = self.r_f[top + 1] * self.dth * self.dz       # (nz,) face areas, mm^2
        for (ch, pol), m in self.pin_masks.items():
            if ch != channel:
                continue
            areas = np.where(m, dA[None, :], 0.0)
            tot = areas.sum()
            cells = self.idx[top][m]
            b[cells] += pol * current_A * areas[m] / tot
        return b

    # -- solve -------------------------------------------------------------
    def solve(self, b, tol=1e-9, maxiter=200):
        """Solve the singular Neumann system with classical AMG + CG."""
        import pyamg
        if getattr(self, "_ml", None) is None:
            self._ml = pyamg.ruge_stuben_solver(self.A, max_coarse=500)
        b = b - b.mean()
        res = []
        x = self._ml.solve(b, tol=tol, accel="cg", maxiter=maxiter, residuals=res)
        self.last_residuals = res
        return x - x.mean()

    def to_field(self, x):
        f = np.full((self.nr, self.nt, self.nz), np.nan)
        f[self.active] = x
        return f

    # -- sampling ----------------------------------------------------------
    def sample_line(self, x, r0, theta0, z):
        """Potential (V) along a line at constant (r, theta); z in mm.

        Sampled on the FEM z-grid and resampled with a natural cubic spline, so
        that the second axial difference seen by the axon stays smooth.
        """
        f = self.to_field(x)
        f = np.where(np.isnan(f), 0.0, f)
        th_ext = np.concatenate([[self.th_c[-1] - 2 * np.pi], self.th_c,
                                 [self.th_c[0] + 2 * np.pi]])
        f_ext = np.concatenate([f[:, -1:, :], f, f[:, :1, :]], axis=1)
        zc = self.z_c
        pts = np.column_stack([np.full(zc.size, r0), np.full(zc.size, theta0), zc])
        v_grid = trilinear((self.r_c, th_ext, zc), f_ext, pts)
        return CubicSpline1D(zc, v_grid)(np.asarray(z, float))

    def electrode_impedance(self, x, channel, current_A=1e-3):
        """Terminal-to-terminal resistance of one channel (ohm)."""
        gel_i = self.term_ring
        v = {}
        for (ch, pol), m in self.pin_masks.items():
            if ch != channel:
                continue
            v[pol] = x[self.idx[gel_i][m]].mean()
        return (v[+1] - v[-1]) / current_A
