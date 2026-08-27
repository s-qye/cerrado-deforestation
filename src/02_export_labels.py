# step 2: turning MapBiomas 30m land cover into 1 km labels

# for each year and each grid cell, 2 numbers are computed:
# native_YYYY: share of the cell that had native vegetation
# cleared_YYYY: share of the cell that went from native to human use

import json
from pathlib import Path
import ee

EE_PROJECT = "cerrado-deforestation"

ASSET = "projects/mapbiomas-public/assets/brazil/lulc/v1"
COLLECTION_ID = 10.0
VERSION = "v1"

START_YEAR = 2008
END_YEAR = 2024

DRIVE_FOLDER = "cerrado_exports"

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "data" / "grid_meta.json"

# cerrado's native vegetation classes
# 3: forest formation
# 4: savanna formation
# 11: wetland formation
# 12: grassland formation
# 29: rocky outcrop
NATIVE = [3, 4, 11, 12, 29]

# human use classes
# 15: pasture
# 21: mosaic of uses
# 20: sugarcane
# 62: cotton
# 46: coffee
# 35: palm oil
# 19: temporary crop
# 30: mining
# 9: forest plantation
# 39: soybean
# 40: rice
# 41: other temporary crops
# 47: citrus
# 48: other perennial crops
# 36: perennial crop
# 24: urban area
HUMAN = [15, 21, 20, 62, 46, 35, 19, 30, 9, 39, 40, 41, 47, 48, 36, 24]

def load_meta():
    """read grid definition"""
    with open(META_PATH) as f:
        meta = json.load(f)
    print(f"grid: {meta['cells_kept']:,} cells, "
          f"{meta['grid_rows']} rows by {meta['grid_cols']} cols, ")
    return meta

def year_image(year):
    """get MapBiomas classification for 1 year as a single band image"""
    return (
        ee.ImageCollection(ASSET)
        .filter(ee.Filter.eq("collection_id", COLLECTION_ID))
        .filter(ee.Filter.eq("version", VERSION))
        .filter(ee.Filter.eq("year", year))
        .first()
        .select("classification")
    )

def build_stack():
    """build 1 image with a native and a cleared band for every year.
    each band has either 0 or 1 at 30m. Averaging 0/1 over a 1 km cell later gives the share of that cell."""
    bands = []
    
    previous = year_image(START_YEAR)
    prev_native = previous.remap(NATIVE, [1] * len(NATIVE), 0)
    bands.append(prev_native.rename(f"native_{START_YEAR}"))

    for year in range(START_YEAR + 1, END_YEAR + 1):
        current = year_image(year)
        cur_native = current.remap(NATIVE, [1] * len(NATIVE), 0)
        cur_human = current.remap(HUMAN, [1] * len(HUMAN), 0)

        # cleared = native last year AND human this year
        cleared = prev_native.And(cur_human)

        bands.append(cur_native.rename(f"native_{year}"))
        bands.append(cleared.rename(f"cleared_{year}"))

        prev_native = cur_native

    return ee.Image.cat(bands).toFloat()

def to_grid(image, meta, native_proj):
    """average the 30m values up to the 1km grid from 01_build_grid.py using 2 steps.
    First go from 30m WGS84 to 250m in the original projection and then go from 250m to 1km while switching to UTM."""

    coarse = (
        image
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
        .reproject(crs=native_proj.atScale(250))
    )
    return (
        coarse
        .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=256)
        .reproject(crs=meta["crs"], crsTransform=meta["ee_crs_transform"])
    )

def main():
    ee.Initialize(project=EE_PROJECT)
    meta = load_meta()

    transform = meta["ee_crs_transform"]
    cols, rows = meta["grid_cols"], meta["grid_rows"]

    x0, y1 = transform[2], transform[5]
    cell = transform[0]
    region = ee.Geometry.Rectangle(
        [x0, y1 - rows * cell, x0 + cols * cell, y1],
        proj=meta["crs"],
        geodesic=False,
    )

    print("Building the band stack...")
    stack = build_stack()
    native_proj = year_image(START_YEAR).projection()
    gridded = to_grid(stack, meta, native_proj)

    print(f"Bands: {len(stack.bandNames().getInfo())}")

    task = ee.batch.Export.image.toDrive(
        image=gridded,
        description=f"cerrado_labels_{START_YEAR}_{END_YEAR}",
        folder=DRIVE_FOLDER,
        fileNamePrefix=f"cerrado_labels_{START_YEAR}_{END_YEAR}",
        crs=meta["crs"],
        crsTransform=transform,
        region=region,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()

    print(f"\ntask started: {task.id}")
    print("Watch it at https://code.earthengine.google.com/tasks")
    print(f"file is in the Drive folder '{DRIVE_FOLDER}'")

if __name__ == "__main__":
    main()