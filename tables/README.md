# Tables

Result tables in publication form. These are **outputs**, not inputs: the notebook reads the
`*.csv` files in the repository root (`sweep_thresholds.csv`, `sweep_runs.csv`,
`block_thresholds.csv`, `cancellation_vs_offset.csv`, `mrg_validation.csv`) and these files
are the formatted versions of the same results.

| file | content |
|---|---|
| `table_threshold_vs_frequency.csv` | activation threshold (mA) for TSINS, TIS and the biphasic control at 3–10 kHz, nerve 5 mm deep |
| `table_threshold_vs_depth.csv` | activation threshold vs nerve depth, 5–25 mm |
| `table_recruitment.csv` | propagated action potentials per second vs channel amplitude, per waveform and carrier frequency |
| `table_block_thresholds.csv` | kilohertz conduction-block threshold vs frequency, single channel and antiphase pair |
| `inclusion_effects.csv` | effect of including the nerve as a resistive inclusion, and of the gel layer, on the potential at the axon |
| `offset_thresholds.csv` | TSINS threshold and residual `fb` activating function as the electrode matrix is offset 0–5 mm from the nerve |
