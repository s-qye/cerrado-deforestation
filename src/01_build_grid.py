# step 1
# build the 1 km analysis grid for 8 municipalities in western Bahia

# outputs:
# data/municipalities.gpkg - the 8 municipal polygons, EPSG:31983
# data/grid_1km.gpkg - the analysis grid, EPSG:31983
# data/grid_meta.json - cell count and the Earth Engine crsTransform

import json
import unicodedata
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import box

import geobr


CRS = "EPSG:31983"
CELL = 1000

TARGETS = [
    "Formosa do Rio Preto",
    "Sao Desiderio",
    "Correntina",
    "Jaborandi",
    "Barreiras",
    "Luis Eduardo Magalhaes",
    "Riachao das Neves",
    "Cocos",
]

OUT = Path(__file__).resolve().parent.parent / "data"

def strip_accents(text):
    """Remove accents so name matching doesnt depend on encoding"""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()

def load_municipalities():
    """Get Bahia municipalities from IGBE and keep the 8 targets"""
    bahia = geobr.read_municipality(code_muni="BA", year=2022) #geobr???
    bahia['key'] = bahia['name_muni'].map(strip_accents)

    wanted = {strip_accents(name) for name in TARGETS}
    muns = bahia[bahia["key"].isin(wanted)].copy()

    found = set(muns['key'])
    missing = wanted - found
    if missing:
        raise ValueError(f"Missing municipalities: {missing}")
    
    muns = muns.to_crs(CRS) #crs???
    return muns[["code_muni", "name_muni", "geometry"]].reset_index(drop=True)

def build_grid(muns, cell=CELL):
    """build a fishnet over the municipal block and keep the inside cells
    
    a cell belongs to a municipality if its centroid is inside the municipal polygon.
    1 municipality per cell and no overlapping cells.
    """
    xmin, ymin, xmax, ymax = muns.total_bounds

    # Snap the origin outward to a clean multiple of the cell size. why??? is this a common practice?
    x0 = np.floor(xmin / cell) * cell
    y0 = np.floor(ymin / cell) * cell
    x1 = np.ceil(xmax / cell) * cell
    y1 = np.ceil(ymax / cell) * cell

    ncol = int(round((x1 - x0) / cell))
    nrow = int(round((y1 - y0) / cell))

    cc, rr = np.meshgrid(np.arange(ncol), np.arange(nrow))
    cc, rr = cc.ravel(), rr.ravel()

    left = x0 + cc * cell
    bottom = y0 + rr * cell
    geoms = [box(l, b, l + cell, b + cell) for l, b in zip(left, bottom)]

    grid = gpd.GeoDataFrame({"row": rr, "col": cc}, geometry=geoms, crs=CRS)

    centroids = grid.copy()
    centroids["geometry"] = grid.geometry.centroid
    joined = gpd.sjoin(
        centroids,
        muns[["code_muni", "name_muni", "geometry"]],
        how="inner",
        predicate="within",
    )

    grid = grid.loc[joined.index].copy()
    grid["code_muni"] = joined["code_muni"].values
    grid["name_muni"] = joined["name_muni"].values
 
    grid = grid.sort_values(["row", "col"]).reset_index(drop=True)
    grid["cell_id"] = np.arange(len(grid), dtype=int)
    grid = grid[["cell_id", "row", "col", "code_muni", "name_muni", "geometry"]]

    transform = [cell, 0, float(x0), 0, -cell, float(y1)]

    meta = {
        "crs": CRS,
        "cell_size_m": cell,
        "grid_rows": nrow,
        "grid_cols": ncol,
        "cells_kept": int(len(grid)),
        "ee_crs_transform": transform,
    }

    return grid, meta

def main():
    OUT.mkdir(exist_ok=True)
 
    print("Fetching municipal boundaries from IBGE...")
    muns = load_municipalities()
    muns.to_file(OUT / "municipalities.gpkg", driver="GPKG")
    area_km2 = muns.geometry.area.sum() / 1e6
    print(f"  {len(muns)} municipalities, {area_km2:,.0f} km2")
 
    print("Building the 1 km grid...")
    grid, meta = build_grid(muns)
    grid.to_file(OUT / "grid_1km.gpkg", driver="GPKG")
 
    with open(OUT / "grid_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"  {meta['cells_kept']:,} cells kept")
    print(f"  full extent {meta['grid_rows']} rows by {meta['grid_cols']} cols")
    print(f"  crsTransform {meta['ee_crs_transform']}")
    print("\nPer municipality:")
    print(grid["name_muni"].value_counts().to_string())
 
 
if __name__ == "__main__":
    main()