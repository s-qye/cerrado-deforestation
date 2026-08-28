# add feature columns to the panel

import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage

START_YEAR = 2008
END_YEAR = 2024

CLEARED_THRESHOLD = 0.10

CELL_KM = 1.0

ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "grid_1km.gpkg"
META_PATH = ROOT / "data" / "grid_meta.json"
PANEL_PATH = ROOT / "data" / "processed" / "panel_labels.parquet"
OUT_PATH = ROOT / "data" / "processed" / "panel_features.parquet"

def to_array(values, rows, cols, nrow, ncol, fill=0.0):
    arr = np.full((nrow, ncol), fill, dtype=float)
    arr[rows, cols] = values
    return arr

def year_arrays(panel, grid, nrow, ncol):
    lookup = grid.set_index("cell_id")[["row", "col"]]
    arrays = {}
    for year, chunk in panel.groupby("year"):
        pos = lookup.loc[chunk["cell_id"]]
        arrays[year] = to_array(
            chunk["cleared_frac"].to_numpy(),
            pos["row"].to_numpy(),
            pos["col"].to_numpy(),
            nrow, ncol,
        )
    return arrays

def build_features(panel, grid, nrow, ncol):
    """Compute 1 feature frame per year to stack them later"""
    arrays = year_arrays(panel, grid, nrow, ncol)
    years = sorted(arrays)

    g_rows = grid["row"].to_numpy()
    g_cols = grid["col"].to_numpy()

    #running total of clearing
    cumulative = np.zeros((nrow, ncol), dtype=float)

    frames = []
    for year in years:
        prev1 = arrays.get(year - 1)
        prev2 = arrays.get(year - 2)
        
        if prev1 is None:
            cumulative = cumulative + arrays[year]
            continue

        # distance to last year's clearing
        mask_prev = prev1 > CLEARED_THRESHOLD
        if mask_prev.any():
            dist_prev = ndimage.distance_transform_edt(~mask_prev) * CELL_KM
        else:
            dist_prev = np.full((nrow, ncol), np.nan)
        
        # distance to the standing frontier aka anything cleared before now
        mask_front = cumulative > CLEARED_THRESHOLD
        if mask_front.any():
            dist_front = ndimage.distance_transform_edt(~mask_front) * CELL_KM
        else:
            dist_front = np.full((nrow, ncol), np.nan)

        # neighborhood pressure
        nbr3 = ndimage.uniform_filter(prev1, size=3, mode="constant", cval=0.0)
        nbr9 = ndimage.uniform_filter(prev1, size=9, mode="constant", cval=0.0)

        frame = pd.DataFrame({
            "cell_id": grid["cell_id"].to_numpy(),
            "year": year,
            "dist_prior_clear_km": dist_prev[g_rows, g_cols],
            "dist_frontier_km": dist_front[g_rows, g_cols],
            "nbr_cleared_3": nbr3[g_rows, g_cols],
            "nbr_cleared_9": nbr9[g_rows, g_cols],
            "cleared_lag1": prev1[g_rows, g_cols],
            "cleared_lag2": (prev2[g_rows, g_cols] if prev2 is not None else np.nan),
            "cum_cleared": cumulative[g_rows, g_cols],
        })
        frames.append(frame)

        cumulative = cumulative + arrays[year]

    return pd.concat(frames, ignore_index=True)

def sanity_checks(df):
    print("\nSanity checks")
    print(f"Rows: {len(df):,}")
    print(f"Years: {df['year'].min()} to {df['year'].max()}")
 
    print("\nMissing values per column:")
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            print(f"  {col}: {n:,}")
 
    print("\nDistance to last year's clearing:")
    d = df["dist_prior_clear_km"].dropna()
    for q in [0.10, 0.50, 0.90, 0.99]:
        print(f"  {q:.0%} of cells within {d.quantile(q):>6.1f} km")
    print(f"  max {d.max():.1f} km")
 
    print("\nMean clearing by distance to last year's clearing:")
    bins = [0, 1, 2, 5, 10, 20, 50, 1000]
    grouped = df.groupby(pd.cut(df["dist_prior_clear_km"], bins))["cleared_frac"]
    for interval, mean in grouped.mean().items():
        n = grouped.size()[interval]
        print(f"  {str(interval):>12}: {mean:.4f}  (n={n:,})")

def main():
    print("loading grid and panel ...")
    grid = gpd.read_file(GRID_PATH)[["cell_id", "row", "col", "name_muni"]]
    panel = pd.read_parquet(PANEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    
    nrow, ncol = meta["grid_rows"], meta["grid_cols"]
    print(f"{len(panel):,} panel rows, {len(grid):,} cells, grid {nrow} x {ncol}")

    print("building features...")
    feats = build_features(panel, grid, nrow, ncol)

    out = panel.merge(feats, on=["cell_id", "year"], how="inner")
    out = out.merge(grid[["cell_id", "name_muni"]], on="cell_id", how="left")

    sanity_checks(out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.relative_to(ROOT)} ({len(out):,} rows)")

if __name__ == "__main__":
    main()