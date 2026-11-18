# `tsins_model` — simulation package

Pure Python/NumPy with two compiled/accelerated dependencies (`numba` for the axon,
`pyamg` for the multigrid preconditioner). Imported by relative path from the repository
root; there is no `setup.py` / `pip install -e .` step.

```python
import sys, os
sys.path.insert(0, os.getcwd())
from tsins_model import volume as V, tissue as T, mrg as M, waveforms as W, \
                        coupling as C, sweep as SW, blocktest as BT
```

## Modules

| module | purpose | public API |
|---|---|---|
| `tissue.py` | Tissue dielectric properties (Gabriel 4-Cole-Cole) used by the TSINS model. | `cole_cole`, `default_sigmas` |
| `volume.py` | Three-layer cylindrical forearm volume conductor for the TSINS / TIS model. | `Geometry`, `contact_centres`, `pin_centres`, `make_grid`, `VolumeConductor` |
| `spectral.py` | Semi-analytic reference solution for the layered cylinder. | `LayeredCylinder` |
| `numerics.py` | Small numerical helpers (scipy.interpolate/optimize are unavailable here). | `natural_cubic_coeffs`, `CubicSpline1D`, `trilinear`, `find_threshold`, `upper_block_threshold` |
| `mrg.py` | McIntyre-Richardson-Grill (2002) double-cable myelinated axon. | `rates`, `steady_gates`, `leak_reversal`, `MRGAxon`, `simulate`, `count_spikes_all_nodes`, `conduction_velocity` |
| `waveforms.py` | Stimulation waveform synthesis: TSINS (Osorio et al., Eqs. 1-6), TIS and the rectangular biphasic control. | `tsins_channels`, `tis_channels`, `biphasic_channels`, `split_bands`, `make_drive`, `channel_sum`, `delivered_charge` |
| `coupling.py` | Couple the finite-element lead fields to the MRG axon. | `LeadFields`, `extracellular_potential`, `residual_at`, `reconstruction_metrics` |
| `sweep.py` | Parameter sweeps over the TSINS / TIS / biphasic conditions. | `run_one`, `threshold` |
| `blocktest.py` | Explicit kilohertz conduction-block test. | `block_run`, `block_threshold` |

## Physics, briefly

**`tissue`** evaluates the Gabriel 4-Cole-Cole dispersion model for muscle, fat, nerve and
skin. TSINS occupies two widely separated bands (a 8–34 Hz harmonic comb and a 3–10 kHz
blocker), and the conductivities differ between them, so the lead fields are solved twice —
once per band. `default_sigmas(f)` returns the conductivity set at frequency `f`; it uses a
prepared/wet skin value (0.10 S/m) by default because the Gabriel dry-skin value implies a
contact impedance the DS5 stimulator cannot drive (`wet_skin=False` restores it).

**`volume`** discretises `div(sigma grad phi) = 0` cell-centred on a graded `(r, theta, z)`
grid: a 25 mm radius, 200 mm long cylinder with muscle `r < 20.5 mm`, fat `20.5–23.5 mm`,
skin `23.5–25 mm`, a 0.5 mm gel layer under each contact, and a 3 mm nerve inclusion at
depth `d`. Insulating everywhere except the gel top faces, where terminal current is
injected. Solved with classical algebraic multigrid preconditioned CG (`pyamg`).

**`spectral`** is an independent check on `volume`: a semi-analytic Bessel-series solution
for the same layered cylinder without the nerve or gel. The two agree to 0.45 % NRMSE.

**`mrg`** is the full McIntyre–Richardson–Grill double cable — node, MYSA, FLUT and six STIN
compartments per internode, with the periaxonal space and myelin sheath as a second cable.
Backward Euler on the cable, exponential Euler on the gates, a 2×2 block-tridiagonal solve
per step, numba-compiled and GIL-free. `balance_leak=True` sets the leak reversal so the node
sits in exact steady state at −80 mV.

**`waveforms`** synthesises the two channel currents for TSINS, TIS and the biphasic control,
splits them into the low-frequency and kilohertz bands (`split_bands`) so each can be
weighted by its own lead field, and reports delivered charge.

**`coupling`** loads the precomputed lead fields (`LeadFields`) and maps channel currents to
the extracellular potential at every compartment of the axon.

**`sweep.threshold`** runs an ascending geometric scan followed by a bracketed bisection.
A plain bisection is wrong here: kHz stimulation is **not monotone** in amplitude — an axon
that fires at threshold can be conduction-blocked at higher amplitudes.

**`blocktest`** implements the explicit block protocol: launch a train of test action
potentials at the proximal node, count how many arrive distally while a kilohertz carrier
is applied.
