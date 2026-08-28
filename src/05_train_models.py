from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
 
ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / "data" / "processed" / "panel_features.parquet"
OUT_DIR = ROOT / "outputs"
 
# A cell counts as cleared if it lost more than this share.
TARGET_THRESHOLD = 0.10
 
TRAIN_END = 2020
VAL_YEAR = 2021
TEST_START = 2022
 
FEATURES = [
    "dist_prior_clear_km",
    "dist_frontier_km",
    "nbr_cleared_3",
    "nbr_cleared_9",
    "cleared_lag1",
    "cleared_lag2",
    "cum_cleared",
    "native_prev",
]
 
 
def top_k_capture(y_true, scores, k):
    """Share of real clearing events found in the riskiest k of cells.
    Sort every cell by predicted risk. Take the top k share. Count how
    many of the year's actual clearing events are in there.
    """
    n = max(1, int(round(len(scores) * k)))
    order = np.argsort(-scores, kind="stable")[:n]
    caught = y_true[order].sum()
    total = y_true.sum()
    return caught / total if total else np.nan
 
 
def evaluate(name, y_true, scores):
    """Grade one set of predictions."""
    return {
        "model": name,
        "base_rate": y_true.mean(),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "capture_top_1pct": top_k_capture(y_true, scores, 0.01),
        "capture_top_5pct": top_k_capture(y_true, scores, 0.05),
        "capture_top_10pct": top_k_capture(y_true, scores, 0.10),
    }
 
 
def main():
    print("Loading features...")
    df = pd.read_parquet(FEAT_PATH)
    df["target"] = (df["cleared_frac"] > TARGET_THRESHOLD).astype(int)
 
    train = df[df["year"] <= TRAIN_END]
    val = df[df["year"] == VAL_YEAR]
    test = df[df["year"] >= TEST_START]
 
    print(f"\nTrain {train['year'].min()}-{train['year'].max()}: "
          f"{len(train):,} rows, {train['target'].mean():.2%} positive")
    print(f"Val   {VAL_YEAR}: {len(val):,} rows, {val['target'].mean():.2%} positive")
    print(f"Test  {test['year'].min()}-{test['year'].max()}: "
          f"{len(test):,} rows, {test['target'].mean():.2%} positive")
 
    y_test = test["target"].to_numpy()
    results = []
 
    # baseline
    base_scores = -test["dist_prior_clear_km"].fillna(1e6).to_numpy()
    results.append(evaluate("Baseline: distance only", y_test, base_scores))
 
    # logistic regression
    medians = train[FEATURES].median()
    X_train = train[FEATURES].fillna(medians)
    X_test = test[FEATURES].fillna(medians)
 
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(scaler.transform(X_train), train["target"])
    logit_scores = logit.predict_proba(scaler.transform(X_test))[:, 1]
    results.append(evaluate("Logistic regression", y_test, logit_scores))
 
    # lightgbm
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train[FEATURES], train["target"],
        eval_set=[(val[FEATURES], val["target"])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_scores = model.predict_proba(test[FEATURES])[:, 1]
    results.append(evaluate("LightGBM", y_test, lgb_scores))
    print(f"\nLightGBM stopped at {model.best_iteration_} trees.")
 
    # report
    res = pd.DataFrame(results)
    print("\n--- Results on the test years ---")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
 
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
 
    print("\n--- What LightGBM leaned on ---")
    print(imp.to_string(index=False))
 
    OUT_DIR.mkdir(exist_ok=True)
    res.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    imp.to_csv(OUT_DIR / "feature_importance.csv", index=False)
 
    # Save test-set predictions
    preds = test[["cell_id", "year"]].copy()
    preds["risk"] = lgb_scores
    preds["target"] = y_test
    preds.to_parquet(ROOT / "data" / "processed" / "test_predictions.parquet",
                     index=False)
 
    print(f"\nSaved results to {OUT_DIR.relative_to(ROOT)}/")
 
 
if __name__ == "__main__":
    main()