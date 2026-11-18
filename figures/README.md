# Figures

Publication figures produced by the notebook, plus the trace data behind them.

| file | content |
|---|---|
| `fig1_model.png` | geometry cross-section, cross-paired electrode layout, channel-1 lead field vs depth, finite-volume vs semi-analytic agreement (0.45 % NRMSE), MRG conduction-velocity validation, strength–duration curve |
| `fig2_reconstruction.png` | channel currents and their in-tissue sum, reconstruction of the harmonic target at the axon, spectrum showing the 8 kHz component cancelled to 2 × 10⁻¹¹ mV, bidirectional propagation, spike train over one 2 Hz period, published on-axis geometry |
| `fig3_carrier.png` | TSINS is carrier-independent (0.83 mA at every `fb`), TIS is not (×2.4 from 3 to 10 kHz); recruitment curves for both |
| `fig4_block.png` | single-channel block threshold vs frequency, arrival of test action potentials under three carriers, tolerance to imperfect cancellation as the matrix is offset |
| `fig5_depth.png` | activation threshold vs nerve depth for all waveforms, normalised depth penalty, and why depth matters: the activating function falls 21-fold from 5 to 25 mm while the potential falls only 2.8-fold |
| `fig_traces.npz` | the raw traces plotted in the figures (26 MB), kept so the panels can be redrawn without re-running the simulations |

`fig_traces.npz` is not read by `TSINS_model.ipynb` — the notebook regenerates what it plots.
It is archived here for redrawing the figures exactly as published.
