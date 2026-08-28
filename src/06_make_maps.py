import json
from pathlib import Path
 
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 
ROOT = Path(__file__).resolve().parent.parent
GRID_PATH = ROOT / "data" / "grid_1km.gpkg"
MUNI_PATH = ROOT / "data" / "municipalities.gpkg"
META_PATH = ROOT / "data" / "grid_meta.json"
PRED_PATH = ROOT / "data" / "processed" / "test_predictions.parquet"
FEAT_PATH = ROOT / "data" / "processed" / "panel_features.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
 
MAP_YEAR = 2024
TARGET_THRESHOLD = 0.10
 
# Share of cells to show on the interactive map
INTERACTIVE_TOP_PCT = 0.05
 
 
def to_array(values, rows, cols, nrow, ncol):
    """Scatter one value per cell into a (row, col) array.
 
    Cells outside the study area stay NaN so they draw as blank.
    """
    arr = np.full((nrow, ncol), np.nan, dtype=float)
    arr[rows, cols] = values
    return arr
 
 
def map_extent(meta):
    """Corner coordinates for imshow, taken from the grid transform."""
    cell, _, x0, _, _, y1 = meta["ee_crs_transform"]
    nrow, ncol = meta["grid_rows"], meta["grid_cols"]
    return [x0, x0 + ncol * cell, y1 - nrow * cell, y1]
 
 
def plot_risk_vs_actual(preds, grid, munis, meta):
    """Two panels: what the model expected, and what happened."""
    year = preds[preds["year"] == MAP_YEAR]
    if year.empty:
        print(f"No predictions for {MAP_YEAR}, skipping the map.")
        return
 
    pos = grid.set_index("cell_id").loc[year["cell_id"], ["row", "col"]]
    rows = pos["row"].to_numpy()
    cols = pos["col"].to_numpy()
    nrow, ncol = meta["grid_rows"], meta["grid_cols"]
 
    risk = to_array(year["risk"].to_numpy(), rows, cols, nrow, ncol)
    actual = to_array(year["target"].to_numpy().astype(float), rows, cols, nrow, ncol)
 
    extent = map_extent(meta)
    boundaries = munis.boundary
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 12), sharex=True, sharey=True,
                             constrained_layout=True)
 
    # Left: predicted risk
    vmax = np.nanpercentile(risk, 99)
    im = axes[0].imshow(risk, origin="lower", extent=extent,
                        cmap="YlOrRd", vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"Predicted risk, {MAP_YEAR}", fontsize=12)
 
    # Right: what got cleared
    actual_cmap = LinearSegmentedColormap.from_list(
        "actual", ["#e8e8e8", "#8B1A1A"])
    axes[1].imshow(actual, origin="lower", extent=extent,
                   cmap=actual_cmap, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Actual clearing over 10 ha, {MAP_YEAR}", fontsize=12)
 
    for ax in axes:
        ax.set_aspect("equal")
        boundaries.plot(ax=ax, color="#333333", linewidth=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
 
    fig.colorbar(im, ax=axes.tolist(), orientation="horizontal",
                 fraction=0.03, pad=0.01, shrink=0.5, label="model score")
    fig.suptitle("Western Bahia deforestation risk", fontsize=14)
    out = FIG_DIR / f"risk_vs_actual_{MAP_YEAR}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")
 
 
def capture_curve(y_true, scores):
    """For every patrol budget, the share of real clearing found."""
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(y_true[order])
    frac_cells = np.arange(1, len(y_true) + 1) / len(y_true)
    frac_found = hits / y_true.sum()
    return frac_cells, frac_found
 
 
def plot_capture_curve(preds, feats):
    """Model against baseline against random guessing."""
    merged = preds.merge(
        feats[["cell_id", "year", "dist_prior_clear_km"]],
        on=["cell_id", "year"], how="left",
    )
 
    y = merged["target"].to_numpy()
    model_x, model_y = capture_curve(y, merged["risk"].to_numpy())
    base_x, base_y = capture_curve(
        y, -merged["dist_prior_clear_km"].fillna(1e6).to_numpy())
 
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(model_x * 100, model_y * 100, color="#B5471B",
            linewidth=2, label="LightGBM")
    ax.plot(base_x * 100, base_y * 100, color="#2F6F4E",
            linewidth=2, linestyle="--", label="Distance to last year's clearing")
    ax.plot([0, 100], [0, 100], color="#999999",
            linewidth=1, linestyle=":", label="Random")
 
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 70)
    ax.set_xlabel("Share of cells inspected (%)")
    ax.set_ylabel("Share of clearing found (%)")
    ax.set_title("How much clearing you find for a given patrol budget")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
 
    for budget in [0.01, 0.05, 0.10]:
        idx = int(len(model_x) * budget) - 1
        ax.plot(budget * 100, model_y[idx] * 100, "o",
                color="#B5471B", markersize=5)
        ax.annotate(f"{model_y[idx]:.0%}",
                    (budget * 100, model_y[idx] * 100),
                    textcoords="offset points", xytext=(6, -3), fontsize=9)
 
    fig.tight_layout()
    out = FIG_DIR / "capture_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out.relative_to(ROOT)}")
 
    for budget in [0.01, 0.05, 0.10]:
        i = int(len(model_x) * budget) - 1
        j = int(len(base_x) * budget) - 1
        print(f"  top {budget:.0%}: model {model_y[i]:.1%}, "
              f"baseline {base_y[j]:.1%}")
 
 
def make_interactive(preds, grid):
    """A zoomable map of the riskiest cells in the final year."""
    try:
        import folium  # noqa: F401
    except ImportError:
        print("folium not installed, skipping the interactive map.")
        return
 
    year = preds[preds["year"] == MAP_YEAR]
    if year.empty:
        return
 
    cutoff = year["risk"].quantile(1 - INTERACTIVE_TOP_PCT)
    top = year[year["risk"] >= cutoff]
 
    gdf = grid.merge(top, on="cell_id", how="inner").to_crs("EPSG:4326")
    gdf["outcome"] = np.where(gdf["target"] == 1, "cleared", "not cleared")
 
    m = gdf[["geometry", "risk", "outcome", "name_muni"]].explore(
        column="risk", cmap="YlOrRd", legend=True,
        tooltip=["risk", "outcome", "name_muni"],
        style_kwds={"weight": 0, "fillOpacity": 0.7},
        tiles="CartoDB positron",
    )
    out = FIG_DIR / "risk_map_interactive.html"
    m.save(str(out))
    print(f"Saved {out.relative_to(ROOT)} "
          f"({len(gdf):,} cells, top {INTERACTIVE_TOP_PCT:.0%})")
 
 
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
 
    print("Loading...")
    grid = gpd.read_file(GRID_PATH)
    munis = gpd.read_file(MUNI_PATH)
    preds = pd.read_parquet(PRED_PATH)
    feats = pd.read_parquet(FEAT_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
 
    print(f"{len(preds):,} predictions, years "
          f"{preds['year'].min()} to {preds['year'].max()}")
 
    plot_risk_vs_actual(preds, grid, munis, meta)
    plot_capture_curve(preds, feats)
    make_interactive(preds, grid)
 
 
if __name__ == "__main__":
    main()