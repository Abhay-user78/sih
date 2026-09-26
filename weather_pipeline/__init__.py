"""Hybrid AI Pipeline for Spatio-Temporal Tracking & Downscaling of
Extreme Weather Anomalies.

Package layout:
  config.py        - global configuration (paths, domain, thresholds)
  download.py      - cdsapi ERA5 download helpers
  load_data.py     - xarray loading + inspection + coordinate handling
  icosahedral.py   - icosphere mesh construction/regrid (Phase 1c, GNN input)
  efi.py           - Extreme Forecast Index implementation (Phase 1d)
  synthetic.py     - PLACEHOLDER dataset generator (Amphan-like) used when
                     real ERA5 credentials are not yet configured
  plot.py          - matplotlib/cartopy helpers
"""

__version__ = "0.1.0"