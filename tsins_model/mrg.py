"""
McIntyre-Richardson-Grill (2002) double-cable myelinated axon.

Geometry per internode:  NODE - MYSA - FLUT - 6 x STIN - FLUT - MYSA - NODE
Two cables: the axolemma (Vm = Vi - Vp) and the periaxonal space (Vp), with the
myelin sheath (Gmy, Cmy) between the periaxonal space and the extracellular
potential Ve.  At nodes the sheath is absent, so Vp = Ve.

Units: mV, ms, uF, mS, uA, um, cm^2 for densities.
Integration: backward Euler on the cable, exponential Euler on the gates,
with a 2x2 block-tridiagonal direct solve each step (numba-compiled).
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- parameters
# fiberD: (g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl)
MRG_TABLE = {
    5.7:  (0.605,  3.4, 1.9, 1.9,  3.4,  500, 35,  80),
    7.3:  (0.630,  4.6, 2.4, 2.4,  4.6,  750, 38, 100),
    8.7:  (0.661,  5.8, 2.8, 2.8,  5.8, 1000, 40, 110),
    10.0: (0.690,  6.9, 3.3, 3.3,  6.9, 1150, 46, 120),
    11.5: (0.700,  8.1, 3.7, 3.7,  8.1, 1250, 50, 130),
    12.8: (0.719,  9.2, 4.2, 4.2,  9.2, 1350, 54, 135),
    14.0: (0.739, 10.4, 4.7, 4.7, 10.4, 1400, 56, 140),
    15.0: (0.767, 11.5, 5.0, 5.0, 11.5, 1450, 58, 145),
    16.0: (0.791, 12.7, 5.5, 5.5, 12.7, 1500, 60, 150),
}

RHOA = 0.7e6        # axoplasmic resistivity, ohm.um
MYCM = 0.1          # uF/cm2 per lamella membrane
MYGM = 0.001        # S/cm2 per lamella membrane
SPACE_P1 = 0.002    # um, periaxonal width at node / MYSA
SPACE_P2 = 0.004    # um, at FLUT
SPACE_I = 0.004     # um, at STIN
CM_AX = 2.0         # uF/cm2, axolemma
V_REST = -80.0      # mV

GNABAR, GNAPBAR, GKBAR, GL = 3.0, 0.010, 0.080, 0.007   # S/cm2
ENA, EK = 50.0, -90.0
G_PAS_MYSA, G_PAS_FLUT, G_PAS_STIN = 0.001, 0.0001, 0.0001   # S/cm2
E_PAS = V_REST

NODE, MYSA, FLUT, STIN = 0, 1, 2, 3


# ------------------------------------------------------------------- kinetics
@njit(cache=True, inline="always")
def _vtrap(num, den_arg, scale):
    if abs(den_arg / scale) < 1e-6:
        return num / den_arg * scale if den_arg != 0.0 else 0.0
    return num / (1.0 - np.exp(-den_arg / scale))


@njit(cache=True)
def rates(v):
    """MRG axnode rate constants at 36 C (1/ms)."""
    x = v + 20.4
    am = 6.57 * x / (1.0 - np.exp(-x / 10.3)) if abs(x) > 1e-6 else 6.57 * 10.3
    x = v + 25.7
    bm = 0.304 * (-x) / (1.0 - np.exp(x / 9.16)) if abs(x) > 1e-6 else 0.304 * 9.16
    x = v + 114.0
    ah = 0.34 * (-x) / (1.0 - np.exp(x / 11.0)) if abs(x) > 1e-6 else 0.34 * 11.0
    bh = 12.6 / (1.0 + np.exp(-(v + 31.8) / 13.4))
    x = v + 27.0
    ap = 0.0353 * x / (1.0 - np.exp(-x / 10.2)) if abs(x) > 1e-6 else 0.0353 * 10.2
    x = v + 34.0
    bp = 0.000883 * (-x) / (1.0 - np.exp(x / 10.0)) if abs(x) > 1e-6 else 0.000883 * 10.0
    as_ = 0.3 / (1.0 + np.exp(-(v + 53.0) / 5.0))
    bs = 0.03 / (1.0 + np.exp(-(v + 90.0) / 1.0))
    return am, bm, ah, bh, ap, bp, as_, bs


def steady_gates(v=V_REST):
    am, bm, ah, bh, ap, bp, a_s, bs = rates(v)
    return (am / (am + bm), ah / (ah + bh), ap / (ap + bp), a_s / (a_s + bs))


def leak_reversal(v=V_REST):
    """e_l that makes the node exactly at steady state at v."""
    m, h, p, s = steady_gates(v)
    i_other = (GNABAR * m ** 3 * h * (v - ENA) + GNAPBAR * p ** 3 * (v - ENA)
               + GKBAR * s * (v - EK))
    return v + i_other / GL


# ------------------------------------------------------------------- geometry
class MRGAxon:
    """Builds the compartment chain and the constant conductance/capacitance arrays."""

    def __init__(self, fiber_diam=10.0, n_nodes=101, z_centre=0.0,
                 balance_leak=True, temperature=36.0):
        if fiber_diam not in MRG_TABLE:
            raise ValueError(f"fiber_diam must be one of {sorted(MRG_TABLE)}")
        self.fiberD = fiber_diam
        g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl = MRG_TABLE[fiber_diam]
        self.axonD, self.nodeD, self.paraD1, self.paraD2 = axonD, nodeD, paraD1, paraD2
        self.deltax, self.paralength2, self.nl = deltax, paralength2, nl
        self.nodelength, self.paralength1 = 1.0, 3.0
        self.interlength = (deltax - self.nodelength - 2 * self.paralength1
                            - 2 * paralength2) / 6.0
        self.n_nodes = n_nodes
        self.e_l = leak_reversal() if balance_leak else -90.0
        self.balance_leak = balance_leak
        self._build()

    def _build(self):
        kinds, lengths, dax, space, gpas = [], [], [], [], []
        unit = ([ (MYSA, self.paralength1, self.paraD1, SPACE_P1, G_PAS_MYSA),
                  (FLUT, self.paralength2, self.paraD2, SPACE_P2, G_PAS_FLUT)]
                + [(STIN, self.interlength, self.axonD, SPACE_I, G_PAS_STIN)] * 6
                + [(FLUT, self.paralength2, self.paraD2, SPACE_P2, G_PAS_FLUT),
                   (MYSA, self.paralength1, self.paraD1, SPACE_P1, G_PAS_MYSA)])
        for i in range(self.n_nodes):
            kinds.append(NODE); lengths.append(self.nodelength)
            dax.append(self.nodeD); space.append(SPACE_P1); gpas.append(0.0)
            if i < self.n_nodes - 1:
                for k, L, d, sp, gp in unit:
                    kinds.append(k); lengths.append(L); dax.append(d)
                    space.append(sp); gpas.append(gp)
        self.kind = np.array(kinds, dtype=np.int64)
        self.L = np.array(lengths)                      # um
        self.dax = np.array(dax)                        # um, true axolemma diam
        self.space = np.array(space)                    # um
        self.gpas = np.array(gpas)                      # S/cm2
        self.n = self.kind.size

        # compartment centres along z (um -> mm), centred on the middle node
        edges = np.concatenate([[0.0], np.cumsum(self.L)])
        self.zc = 0.5 * (edges[1:] + edges[:-1])
        self.zc = (self.zc - self.zc[self.node_index[self.n_nodes // 2]]) * 1e-3  # mm
        self.length_mm = edges[-1] * 1e-3

        # --- areas and lumped parameters -------------------------------
        A_ax = np.pi * (self.dax * 1e-4) * (self.L * 1e-4)        # cm^2, axolemma
        A_my = np.pi * (self.fiberD * 1e-4) * (self.L * 1e-4)     # cm^2, myelin
        self.A_ax, self.A_my = A_ax, A_my
        self.Cm = CM_AX * A_ax                                    # uF
        self.Cmy = (MYCM / (2 * self.nl)) * A_my                  # uF
        self.Gmy = (MYGM / (2 * self.nl)) * A_my * 1e3            # mS
        self.Gpas = self.gpas * A_ax * 1e3                        # mS
        is_node = self.kind == NODE
        self.Cmy[is_node] = 0.0
        self.Gmy[is_node] = 0.0
        self.is_node = is_node
        self.gna = GNABAR * A_ax * 1e3 * is_node                  # mS
        self.gnap = GNAPBAR * A_ax * 1e3 * is_node
        self.gk = GKBAR * A_ax * 1e3 * is_node
        self.gl = GL * A_ax * 1e3 * is_node

        # --- axial conductances (mS) between neighbouring compartments --
        half_R_ax = RHOA * (self.L / 2.0) / (np.pi * (self.dax / 2.0) ** 2)   # ohm
        A_peri = np.pi * ((self.dax / 2 + self.space) ** 2 - (self.dax / 2) ** 2)
        half_R_p = RHOA * (self.L / 2.0) / A_peri                             # ohm
        self.Ga = 1e3 / (half_R_ax[:-1] + half_R_ax[1:])          # mS, n-1
        self.Gp = 1e3 / (half_R_p[:-1] + half_R_p[1:])            # mS, n-1

    @property
    def node_index(self):
        return np.where(self.kind == NODE)[0] if hasattr(self, "kind") else None

    def node_positions(self):
        return self.zc[self.node_index]

    def initial_state(self):
        m, h, p, s = steady_gates()
        n = self.n
        return dict(Vm=np.full(n, V_REST), Vp=np.zeros(n),
                    m=np.full(n, m), h=np.full(n, h),
                    p=np.full(n, p), s=np.full(n, s))


# ----------------------------------------------------------------- time loop
@njit(cache=True, fastmath=True, nogil=True)
def _simulate(dt, nstep, Vm, Vp, mg, hg, pg, sg,
              Cm, Cmy, Gmy, Gpas, gna, gnap, gk, gl, e_l, is_node,
              Ga, Gp, lead, ivec, istim_idx, istim, rec_idx, rec_stride,
              spike_node, spike_thresh):
    n = Vm.size
    nk = lead.shape[1]
    A = np.zeros((n, 2, 2)); B = np.zeros((n, 2, 2)); C = np.zeros((n, 2, 2))
    d = np.zeros((n, 2))
    # LU workspace for block-Thomas
    Cp = np.zeros((n, 2, 2)); dp = np.zeros((n, 2))
    Ve = np.zeros(n); Ve_new = np.zeros(n)
    nrec = (nstep // rec_stride) + 1
    rec = np.zeros((nrec, rec_idx.size), dtype=np.float32)
    rec_t = np.zeros(nrec, dtype=np.float32)
    spikes = np.zeros(2000, dtype=np.float64)
    nspk = 0
    armed = True

    for k in range(nk):
        for i in range(n):
            Ve[i] += lead[i, k] * ivec[0, k]
    # start the periaxonal space at the local Ve (quiescent initial condition)
    for i in range(n):
        Vp[i] = Ve[i]

    irec = 0
    for step in range(nstep):
        t = (step + 1) * dt
        for i in range(n):
            Ve_new[i] = 0.0
        for k in range(nk):
            ik = ivec[step + 1, k]
            if ik != 0.0:
                for i in range(n):
                    Ve_new[i] += lead[i, k] * ik

        # --- gates (exponential Euler on Vm at time n) -----------------
        for i in range(n):
            if not is_node[i]:
                continue
            v = Vm[i]
            x = v + 20.4
            am = 6.57 * x / (1.0 - np.exp(-x / 10.3)) if abs(x) > 1e-6 else 6.57 * 10.3
            x = v + 25.7
            bm = 0.304 * (-x) / (1.0 - np.exp(x / 9.16)) if abs(x) > 1e-6 else 0.304 * 9.16
            x = v + 114.0
            ah = 0.34 * (-x) / (1.0 - np.exp(x / 11.0)) if abs(x) > 1e-6 else 0.34 * 11.0
            bh = 12.6 / (1.0 + np.exp(-(v + 31.8) / 13.4))
            x = v + 27.0
            ap = 0.0353 * x / (1.0 - np.exp(-x / 10.2)) if abs(x) > 1e-6 else 0.0353 * 10.2
            x = v + 34.0
            bp = 0.000883 * (-x) / (1.0 - np.exp(x / 10.0)) if abs(x) > 1e-6 else 0.000883 * 10.0
            a_s = 0.3 / (1.0 + np.exp(-(v + 53.0) / 5.0))
            bs = 0.03 / (1.0 + np.exp(-(v + 90.0) / 1.0))
            r = am + bm; mg[i] = am / r + (mg[i] - am / r) * np.exp(-dt * r)
            r = ah + bh; hg[i] = ah / r + (hg[i] - ah / r) * np.exp(-dt * r)
            r = ap + bp; pg[i] = ap / r + (pg[i] - ap / r) * np.exp(-dt * r)
            r = a_s + bs; sg[i] = a_s / r + (sg[i] - a_s / r) * np.exp(-dt * r)

        # --- assemble the block-tridiagonal system ---------------------
        for i in range(n):
            gal = Ga[i - 1] if i > 0 else 0.0
            gar = Ga[i] if i < n - 1 else 0.0
            gpl = Gp[i - 1] if i > 0 else 0.0
            gpr = Gp[i] if i < n - 1 else 0.0
            if is_node[i]:
                gNa = gna[i] * mg[i] * mg[i] * mg[i] * hg[i]
                gNap = gnap[i] * pg[i] * pg[i] * pg[i]
                gK = gk[i] * sg[i]
                gtot = gNa + gNap + gK + gl[i]
                irev = (gNa + gNap) * 50.0 + gK * (-90.0) + gl[i] * e_l
            else:
                gtot = Gpas[i]
                irev = Gpas[i] * (-80.0)
            cdt = Cm[i] / dt
            A[i, 0, 0] = -gal; A[i, 0, 1] = -gal
            C[i, 0, 0] = -gar; C[i, 0, 1] = -gar
            B[i, 0, 0] = cdt + gtot + gal + gar
            B[i, 0, 1] = gal + gar
            d[i, 0] = cdt * Vm[i] + irev
            if i == istim_idx:
                d[i, 0] += istim[step + 1]
            if is_node[i]:
                A[i, 1, 0] = 0.0; A[i, 1, 1] = 0.0
                C[i, 1, 0] = 0.0; C[i, 1, 1] = 0.0
                B[i, 1, 0] = 0.0; B[i, 1, 1] = 1.0
                d[i, 1] = Ve_new[i]
            else:
                cmy = Cmy[i] / dt
                A[i, 1, 0] = -gal; A[i, 1, 1] = -(gal + gpl)
                C[i, 1, 0] = -gar; C[i, 1, 1] = -(gar + gpr)
                B[i, 1, 0] = gal + gar
                B[i, 1, 1] = cmy + Gmy[i] + gal + gar + gpl + gpr
                d[i, 1] = cmy * Vp[i] + cmy * (Ve_new[i] - Ve[i]) + Gmy[i] * Ve_new[i]
                if i == istim_idx:
                    d[i, 1] += istim[step + 1]

        # --- block-Thomas forward sweep --------------------------------
        M = np.zeros((2, 2))
        for i in range(n):
            for a in range(2):
                for b in range(2):
                    M[a, b] = B[i, a, b]
            if i > 0:
                for a in range(2):
                    for b in range(2):
                        s_ = 0.0
                        for c in range(2):
                            s_ += A[i, a, c] * Cp[i - 1, c, b]
                        M[a, b] -= s_
                for a in range(2):
                    s_ = 0.0
                    for c in range(2):
                        s_ += A[i, a, c] * dp[i - 1, c]
                    d[i, a] -= s_
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            i00, i01 = M[1, 1] / det, -M[0, 1] / det
            i10, i11 = -M[1, 0] / det, M[0, 0] / det
            if i < n - 1:
                Cp[i, 0, 0] = i00 * C[i, 0, 0] + i01 * C[i, 1, 0]
                Cp[i, 0, 1] = i00 * C[i, 0, 1] + i01 * C[i, 1, 1]
                Cp[i, 1, 0] = i10 * C[i, 0, 0] + i11 * C[i, 1, 0]
                Cp[i, 1, 1] = i10 * C[i, 0, 1] + i11 * C[i, 1, 1]
            dp[i, 0] = i00 * d[i, 0] + i01 * d[i, 1]
            dp[i, 1] = i10 * d[i, 0] + i11 * d[i, 1]
        # --- back substitution ------------------------------------------
        Vm[n - 1] = dp[n - 1, 0]; Vp[n - 1] = dp[n - 1, 1]
        for i in range(n - 2, -1, -1):
            Vm[i] = dp[i, 0] - Cp[i, 0, 0] * Vm[i + 1] - Cp[i, 0, 1] * Vp[i + 1]
            Vp[i] = dp[i, 1] - Cp[i, 1, 0] * Vm[i + 1] - Cp[i, 1, 1] * Vp[i + 1]

        for i in range(n):
            Ve[i] = Ve_new[i]

        # --- spike detection at one node --------------------------------
        if armed:
            if Vm[spike_node] > spike_thresh:
                if nspk < spikes.size:
                    spikes[nspk] = t
                nspk += 1
                armed = False
        else:
            if Vm[spike_node] < spike_thresh - 10.0:
                armed = True

        if (step + 1) % rec_stride == 0 and irec + 1 < nrec:
            irec += 1
            rec_t[irec] = t
            for j in range(rec_idx.size):
                rec[irec, j] = Vm[rec_idx[j]]
    return rec_t[:irec + 1], rec[:irec + 1], spikes[:min(nspk, spikes.size)], nspk


# --------------------------------------------------------------- driver API
def simulate(axon, lead, ivec, dt, istim=None, istim_idx=-1,
             rec_nodes=True, rec_dt=0.02, spike_node=None, spike_thresh=-20.0,
             state=None):
    """Integrate the axon.

    lead : (n_comp, nk) extracellular potential per unit channel current, in mV/(unit of ivec)
    ivec : (nstep+1, nk) channel drive, same length as the time grid
    istim: (nstep+1,) intracellular current (uA) injected at compartment istim_idx
    """
    n = axon.n
    lead = np.ascontiguousarray(np.atleast_2d(lead).astype(np.float64))
    if lead.shape[0] != n:
        lead = lead.T
    ivec = np.ascontiguousarray(np.atleast_2d(ivec).astype(np.float64))
    if ivec.shape[1] != lead.shape[1]:
        ivec = ivec.T
    nstep = ivec.shape[0] - 1
    st = state if state is not None else axon.initial_state()
    Vm, Vp = st["Vm"].copy(), st["Vp"].copy()
    mg, hg, pg, sg = (st["m"].copy(), st["h"].copy(), st["p"].copy(), st["s"].copy())
    if istim is None:
        istim = np.zeros(nstep + 1)
    nodes = axon.node_index
    rec_idx = nodes.astype(np.int64) if rec_nodes is True else np.asarray(rec_nodes, dtype=np.int64)
    rec_stride = max(1, int(round(rec_dt / dt)))
    sn = int(nodes[len(nodes) // 2]) if spike_node is None else int(spike_node)
    t, rec, spikes, nspk = _simulate(
        dt, nstep, Vm, Vp, mg, hg, pg, sg,
        axon.Cm, axon.Cmy, axon.Gmy, axon.Gpas, axon.gna, axon.gnap, axon.gk,
        axon.gl, axon.e_l, axon.is_node, axon.Ga, axon.Gp,
        lead, ivec, int(istim_idx), np.ascontiguousarray(istim.astype(np.float64)),
        rec_idx, rec_stride, sn, float(spike_thresh))
    return dict(t=t, v=rec, rec_idx=rec_idx, spikes=spikes, n_spikes=int(nspk),
                state=dict(Vm=Vm, Vp=Vp, m=mg, h=hg, p=pg, s=sg))


def count_spikes_all_nodes(v, thresh=-20.0):
    """Spike count per recorded trace (upward threshold crossings)."""
    above = v > thresh
    return (above[1:] & ~above[:-1]).sum(axis=0)


def conduction_velocity(t, v, z_nodes, thresh=-20.0, skip=0.25):
    """Velocity (m/s) from the AP peak times over the middle half of the axon."""
    lo, hi = int(skip * len(z_nodes)), int((1 - skip) * len(z_nodes))
    tt, zz = [], []
    for j in range(lo, hi):
        tr = v[:, j]
        if tr.max() < thresh:
            continue
        tt.append(t[int(np.argmax(tr))]); zz.append(z_nodes[j])
    if len(tt) < 3:
        return np.nan
    tt, zz = np.array(tt), np.array(zz)
    slope = np.polyfit(tt, zz, 1)[0]        # mm/ms
    return abs(slope)                        # mm/ms == m/s
