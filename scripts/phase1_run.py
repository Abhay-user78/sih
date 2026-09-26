"""PHASE 1 driver: load data -> inspect -> icosahedral mesh -> EFI.

STOP AND VERIFY gate: this script must run end-to-end and produce the
processed arrays + plots in data/processed and outputs/ before Phase 2.

Usage:
    python -m scripts.phase1_run

Data source selected by CONFIG.data_source:
    synthetic (default; no credentials needed)
    era5      (requires ~/.cdsapirc + data/raw/era5_amphan_test.nc and a
               climatology baseline in data/processed/era5_climatology.nc;
               see phase1_climatology.py)
"""
from __future__ import annotations

import argparse

import numpy as np

from weather_pipeline.config import CONFIG, DATA_PROCESSED, OUTPUTS
from weather_pipeline.icosahedral import build_mesh, subset_to_bbox, \
    interpolate_field, latlon
from weather_pipeline import efi as efi_mod
from weather_pipeline import plot as plot_mod
from weather_pipeline.synthetic import build_synthetic_amphan, climatology_xr
from weather_pipeline.load_data import open_data


def load_fields():
    """Return (ds, lat, lon) for the current data source."""
    if CONFIG.data_source == "synthetic":
        ds = build_synthetic_amphan(save=True)
        return ds
    ld = open_data()
    return ld.ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timestep", type=int, default=20,
                    help="index of the timestep to analyse (0=16 May 00Z; "
                         "20=21 May 12Z, near landfall)")
    ap.add_argument("--variable", default="geopotential",
                    help="variable used for EFI/mesh demo")
    ap.add_argument("--level", type=int, default=500,
                    help="pressure level (hPa)")
    args = ap.parse_args()

    ds = load_fields()
    lat = ds.latitude.values
    lon = ds.longitude.values
    lon = np.mod(lon, 360.0)
    var = args.variable
    lev = args.level

    print("\n--- [1b] dataset inspection ---")
    for name, da in ds.data_vars.items():
        print(f"  {name:<24} dims=({', '.join(da.dims)}) {da.dtype}")

    da = ds[var].sel(level=lev)
    if "time" in da.dims:
        current = da.isel(time=args.timestep).values
        tlabel = str(da.isel(time=args.timestep).time.values)
    else:
        current = da.values
        tlabel = "static"
    llat, llon = np.meshgrid(lat, lon, indexing="ij")

    # ---------------------------------------------------------------
    print("\n--- [1c] icosahedral mesh ---")
    mesh = build_mesh(level=CONFIG.mesh_level)
    sub = subset_to_bbox(mesh, CONFIG.domain)
    nlan, nlon = latlon(sub["vertices"])
    print(f"  full mesh: {mesh['vertices'].shape[0]} nodes "
          f"{mesh['edge_index'].shape[1]} edges")
    print(f"  bbox subset: {len(nlan)} nodes, {sub['edge_index'].shape[1]} edges")

    node_vals = interpolate_field(current, lat, lon, nlan, nlon, k=4)
    plot_mod.plot_mesh_edges(sub, str(OUTPUTS / "phase1_mesh_graph.png"))
    plot_mod.plot_grid_field(
        lat, lon, current,
        f"{var} {lev} hPa  {tlabel} (source={CONFIG.data_source})",
        str(OUTPUTS / "phase1_grid_field.png"))
    plot_mod.plot_mesh(
        nlan, nlon, node_vals,
        f"{var} {lev} hPa on icosphere ({len(nlan)} nodes)",
        str(OUTPUTS / "phase1_mesh_field.png"))

    # ---------------------------------------------------------------
    print("\n--- [1d] Extreme Forecast Index ---")
    if CONFIG.data_source == "synthetic":
        clim = climatology_xr()
        # baseline dims (year, level, lat, lon); axis=year for EFI
        clim_vals = clim[var].sel(level=lev).values          # (year, lat, lon)
        efi_field = efi_mod.compute_efi(current, clim_vals, axis=0)
        print("  EFI computed against synthetic 30-year climatology")
    else:
        # ERA5 real-data path: requires a downscoped climatology file.
        cp = DATA_PROCESSED / "era5_climatology.nc"
        if not cp.exists():
            raise SystemExit(
                "[era5] climatology baseline missing. Run "
                "phase1_climatology.py after downloading the ERA5 baseline "
                "(30 years x May) via cdsapi.")
        import xarray as xr
        clim = xr.open_dataset(cp)[var].sel(level=lev).values  # (year,lat,lon)
        efi_field = efi_mod.compute_efi(current, clim, axis=0)
        print("  EFI computed against ERA5 climatology")

    node_efi = interpolate_field(efi_field, lat, lon, nlan, nlon, k=4)
    plot_mod.plot_grid_field(
        lat, lon, efi_field,
        f"EFI {var} {lev} hPa {tlabel}",
        str(OUTPUTS / "phase1_efi_grid.png"),
        vmin=-1, vmax=1, cmap="RdBu_r")
    plot_mod.plot_mesh(
        nlan, nlon, node_efi,
        f"EFI on icosphere ({len(nlan)} nodes)",
        str(OUTPUTS / "phase1_efi_mesh.png"),
        cmap="RdBu_r")

    # summary stats
    thr = CONFIG.efi_unusual
    frac_severe = float((np.abs(efi_field) >= thr).mean() * 100)
    print(f"\n  EFI range [{efi_field.min():.3f}, {efi_field.max():.3f}], "
          f"{frac_severe:.1f}% of grid >= |{thr}| (unusual+)")
    hi_i = np.unravel_index(np.argmax(efi_field), efi_field.shape)
    lo_i = np.unravel_index(np.argmin(efi_field), efi_field.shape)
    print(f"  EFI=+{efi_field[hi_i]:.2f} (above normal) near lat "
          f"{lat[hi_i[0]]:.2f}, lon {lon[hi_i[1]]:.2f}")
    print(f"  EFI={efi_field[lo_i]:.2f} (below normal) near lat "
          f"{lat[lo_i[0]]:.2f}, lon {lon[lo_i[1]]:.2f}")
    if CONFIG.data_source == "synthetic":
        print("  NOTE: synthetic cyclone centre moved 89.5E->88.7E toward "
              "lat 22.5 (landfall). Expect EFI near -1 (low geopotential / "
              "storm) following that track.")

    np.savez(
        DATA_PROCESSED / "phase1_processed.npz",
        efi_grid=efi_field, node_efi=node_efi,
        node_lat=nlan, node_lon=nlon,
        node_xyz=sub["vertices"], node_feats=node_vals,
        edge_index=sub["edge_index"], global_index=sub["global_index"],
        current_grid=current, lat=lat, lon=lon,
        variable=var, level=lev, timestep=args.timestep,
        timestep_label=tlabel,
    )
    print("\n  saved data/processed/phase1_processed.npz")
    print("\nPHASE 1 STOP-AND-VERIFY: inspect outputs/phase1_efi_*.png (and")
    print("the printed EFI max location) before Phase 2.")


if __name__ == "__main__":
    main()