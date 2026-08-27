# read the GeoTIFF exported from 02_export_labels.py and turn it into a table
# output is 1 row per cell per year: cell_id, year, native_frac, native_prev, cleared_frac, cleared_ha

from pathlib import Path
from rasterio.transform import rowcol

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import json

START_YEAR = 2008
END_YEAR = 2024

ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "grid_1km.gpkg"
META_PATH = ROOT / "data" / "grid_meta.json"
RASTER_PATH = ROOT / "data" / "raw" / f"cerrado_labels_{START_YEAR}_{END_YEAR}.tif"
OUT_PATH = ROOT / "data" / "processed" / "panel_labels.parquet"

HA_PER_CELL = 100.0

def expected_band_names():
    names = [f"native_{START_YEAR}"]
    for year in range(START_YEAR + 1, END_YEAR + 1):
        names.append(f"native_{year}")
        names.append(f"cleared_{year}")
    return names

def read_bands(path, names):
    """read every band into a dict of 2d arrays. keys = band names"""
    with rasterio.open(path) as src:
        if src.count!= len(names):
            raise ValueError(
                f"Raster has {src.count} bands, expected {len(names)}."
            )
        described = [d for d in src.descriptions if d]
        if described and list(described) != names:
            print(f"Band names in file don't match expected order")
            print(f"file says: {described[:4]}")
            print(f"script expects: {names[:4]}")

        arrays = {name: src.read(i + 1) for i, name in enumerate(names)}
        shape = (src.height, src.width)
        transform = src.transform
    return arrays, shape, transform

def build_panel(grid, arrays, transform, shape):
    """fine each cell's pixel by its map coords"""
    cent = grid.geometry.centroid
    r_idx, c_idx = rowcol(transform, cent.x.to_numpy(), cent.y.to_numpy())
    r_idx, c_idx = np.asarray(r_idx), np.asarray(c_idx)

    h, w = shape
    if not ((0 <= r_idx).all() and (r_idx < h).all()
            and (0 <= c_idx).all() and (c_idx < w).all()):
        raise ValueError("Some cells fall outside the raster. Check the export region.")
 
    frames = []
    for year in range(START_YEAR + 1, END_YEAR + 1):
        native = arrays[f"native_{year}"][r_idx, c_idx]
        prev = arrays[f"native_{year - 1}"][r_idx, c_idx]
        cleared = arrays[f"cleared_{year}"][r_idx, c_idx]
 
        frames.append(pd.DataFrame({
            "cell_id": grid["cell_id"].to_numpy(),
            "year": year,
            "native_frac": native,
            "native_prev": prev,
            "cleared_frac": cleared,
        }))
 
    panel = pd.concat(frames, ignore_index=True)
    panel["cleared_ha"] = panel["cleared_frac"] * HA_PER_CELL
    return panel

def sanity_checks(panel):
    print("\nSanity checks:")
    print(f"Rows: {len(panel):,}")
    print(f"Cells: {panel['cell_id'].nunique():,}")
    print(f"Years: {panel['year'].min()} to {panel['year'].max()}")

    for col in ["native_frac", "native_prev", "cleared_frac"]:
        lo, hi = panel[col].min(), panel[col].max()
        flag = "" if (lo >= -1e-6 and hi <= 1 + 1e-6) else "  <-- OUT OF RANGE"
        print(f"{col}: min {lo:.4f}, max {hi:.4f}{flag}")
 
    # Clearing cannot exceed what was there last year.
    impossible = (panel["cleared_frac"] > panel["native_prev"] + 1e-4).sum()
    print(f"Rows where cleared > previous native: {impossible:,}"
          + ("  <-- PROBLEM" if impossible > 0 else "  (good)"))
 
    print("\nCleared hectares per year:")
    yearly = panel.groupby("year")["cleared_ha"].sum()
    for year, ha in yearly.items():
        print(f"  {year}  {ha:>12,.0f} ha")
 
    print("\nClass balance, cells with more than 1 ha cleared:")
    for thresh, label in [(0.01, "> 1 ha"), (0.10, "> 10 ha")]:
        share = (panel["cleared_frac"] > thresh).mean()
        print(f"  {label:>8}: {share:.4%} of rows")

def main():
    print("Grid loading...")
    grid = gpd.read_file(GRID_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)

    names = expected_band_names()
    print(f"Reading {len(names)} bands from {RASTER_PATH.name}...")
    arrays, shape, transform = read_bands(RASTER_PATH, names)

    expected_shape = (meta["grid_rows"], meta["grid_cols"])
    if shape != expected_shape:
        print(f"Note: raster is {shape}, grid is {expected_shape}. "
              "Cells are matched by coordinate, so extra edge rows are ignored.")


    panel = build_panel(grid, arrays, transform, shape)
    sanity_checks(panel)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.relative_to(ROOT)}")

if __name__ == "__main__":
    main()