TSINS vs TIS simulation model
=============================
Run `TSINS_model.ipynb` from the directory this archive unpacks into
(the notebook imports `tsins_model/` by relative path and reads the
.npz / .csv files from the working directory).

Environment: python >= 3.11 with numpy, scipy, matplotlib, pandas,
numba, pyamg, ipywidgets, jupyterlab.

tsins_model/
  tissue.py     Gabriel 4-Cole-Cole tissue conductivities
  volume.py     3-layer cylindrical finite-volume volume conductor
  spectral.py   semi-analytic layered-cylinder reference solution
  numerics.py   cubic spline, trilinear interpolation, threshold search
  mrg.py        McIntyre-Richardson-Grill double-cable axon (numba)
  waveforms.py  TSINS / TIS / biphasic synthesis
  coupling.py   lead fields -> axon
  sweep.py      parameter sweeps
  blocktest.py  kHz conduction-block protocol

leadfields.npz            aligned matrix, 5 depths x 2 bands x 2 channels
leadfields_offset{1,2,3,5}.npz   matrix offset off the nerve, 5 mm depth

Rebuilding the lead fields from the finite-element model takes about
20 s per solve; set RESOLVE = True in the notebook.
