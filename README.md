# Forecasting deforestation risk in the Brazilian Cerrado

Predicting which 1 km cells in the western Bahia soy frontier will lose native
vegetation in the following year.

**Result:** inspecting the highest-risk 5% of cells finds 22% of the next year's clearing, against 14% using proximity to last year's clearing alone.

![Capture curve](outputs/figures/capture_curve.png)

---

## The question

Can land cover history, clearing proximity, and neighbourhood pressure predict
where native Cerrado vegetation is converted to human use next year, and does an ML model beat a proximity rule?

Deforestation spreads from its own edge, and "distance to last year's clearing" is a strong predictor on its own. A model would have to beat a heuristic to justify its usage.

## Study area

Eight municipalities in western Bahia: Formosa do Rio Preto, São Desidério,
Correntina, Jaborandi, Barreiras, Luís Eduardo Magalhães, Riachão das Neves,
and Cocos.

This is the MATOPIBA agricultural frontier, which is the most active soy expansion zone
in the Cerrado. This was chosen over the Amazon for two reasons: Brazil's Soy Moratorium doesn't cover the Cerrado, and the Forest Code requires only a 20% Legal Reserve here compared with 80% in Amazon forest.

## Data

| Source | What | Notes |
|---|---|---|
| MapBiomas Collection 10, v1 | Annual 30 m land cover, 2008-2024 | Accessed via Google Earth Engine. Past years are relabelled when a new collection ships, so this number matters for reproducibility. |
| IBGE via `geobr` | Municipal boundaries, 2022 vintage | Boundaries change; the year is pinned. |

## Method

1. **Grid:** Divide the study area into 80,390 cells of 1 km², in EPSG:31983
   (UTM 23S). Each cell gets a permanent ID.
2. **Labels:** In Earth Engine, classify each 30 m pixel as native vegetation or
   human use, then mark clearing where a pixel was native in year *t-1* and
   human use in year *t*. Average up to the 1 km grid.
3. **Panel:** One row per cell per year, 2009-2024.
4. **Features:** Distance to last year's clearing, distance to the standing
   frontier, neighbourhood clearing at two scales, own-cell lags, cumulative
   clearing, and remaining native vegetation.
5. **Split:** Train 2010-2020, validate 2021, and test 2022-2024.
6. **Models:** Proximity baseline, logistic regression, LightGBM.

### Target definition

A cell is positive if more than 10% of it (10 ha) was cleared that year.

### Avoiding leakage

**Spatial:** Since neighbouring cells look alike, a random split would put a cell's
neighbor in training and the cell itself in test, letting the model score well
by recognising the neighborhood. This is why splitting by year was chosen.

**Temporal:** `native_frac` and `cleared_frac` describe the end of the target
year, so both would help show the answer. Only backward-looking columns are used.
`native_prev` describes the start of the year and is okay to use.

## Results

Test years 2022-2024. Base rate 3.32%.

| Model | PR-AUC | ROC-AUC | Top 1% | Top 5% | Top 10% |
|---|---|---|---|---|---|
| Baseline: distance only | 0.065 | 0.675 | 3.4% | 14.5% | 26.9% |
| Logistic regression | 0.075 | 0.734 | 0.8% | 15.8% | 30.1% |
| **LightGBM** | **0.100** | 0.773 | **4.8%** | **22.2%** | **35.1%** |

We focus on PR-AUC over ROC-AUC. With positives at 3.3%, ROC-AUC stays high
even for a weak model because negatives dominate the denominator. PR-AUC
compares against the base rate: 0.100 against 0.033 is three times better than
chance.

### Feature Importance

| Feature | Importance |
|---|---|
| native_prev | 202 |
| nbr_cleared_9 | 163 |
| nbr_cleared_3 | 149 |
| cleared_lag2 | 144 |
| dist_prior_clear_km | 131 |
| dist_frontier_km | 113 |
| cleared_lag1 | 98 |
| cum_cleared | 80 |

These are split counts, which favor continuous features with many possible
split points. This isn't to be mistake for a causal ranking but more like just a rough guide.

### Maps

![Risk versus actual](outputs/figures/risk_vs_actual_2024.png)

Predicted risk forms thin loops rather than solid shapes, tracing the edges of
existing cleared land.

Actual clearing arrives in large contiguous blocks, since a property clears at
once. Both maps concentrate activity in the north and centre and thin toward the
south.

## Limitations

**This measures conversion, not illegality:** A pixel that goes from savanna to
soy counts the same whether the clearing was permitted or not. Distinguishing
legal from illegal conversion needs state authorization records from INEMA,
which are not part of this dataset. MapBiomas Alerta, which does have that data,
still treats absence of a permit only as an *indication* of illegality and
leaves the judgement to public agencies. This project makes no such claim.

**Municipal edge effects:** Distance features treat cells outside the eight
municipalities as uncleared. Clearing does not stop at an administrative border,
so cells near the study area boundary get distances that are slightly too large.

**Double aggregation:** Going from 30 m WGS84 straight to 1 km UTM exceeds Earth
Engine's reprojection limit, so the pipeline coarsens to 250 m in the source
projection first. Averaging twice is not identical to averaging once. Expect
errors of a fraction of a percent per cell.

**Rising base rate:** Positives run 2.20% in training, 2.70% in validation, and
3.32% in test. Clearing accelerated over 2022-2024, so the model was trained on a
calmer period than it was tested on.

**One-year horizon only:** Multi-year forecasts would require feeding predictions
back as inputs, and errors compound quickly. Not attempted.

### Validation against published figures

Gross clearing from this pipeline was 91,771 ha in 2009 and 61,900 ha in 2010. MapBiomas Collection 11 municipality statistics give net native vegetation loss of 93,706 ha and 73,670 ha for the same years and area. The pipeline reads lower, as expected, since net change also captures transitions to classes outside the human-use definition. This pipeline used Collection 10.

## Repository

```
src/
  01_build_grid.py       Municipal boundaries -> 80,390 cell grid
  02_export_labels.py    MapBiomas -> 1 km clearing labels (Earth Engine)
  03_load_labels.py      GeoTIFF -> panel table
  04_build_features.py   Spatial and temporal features
  05_train_models.py     Temporal split, baseline, models, metrics
  06_make_maps.py        Risk maps and capture curve
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On macOS, LightGBM needs OpenMP, which pip cannot install:

```bash
brew install libomp
```

Earth Engine requires a registered Cloud project. Authenticate once with
`ee.Authenticate()`, then set `EE_PROJECT` in `src/02_export_labels.py`.