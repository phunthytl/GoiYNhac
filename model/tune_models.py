from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MODEL_DIR, RANDOM_STATE, REPORT_DIR, TARGET
from database import load_training_pairs_df
from model.train_lightgbm import (
    classification_metrics,
    prepare_features,
    precision_recall_ndcg_at_k,
    safe_train_test_split,
)

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "Thiếu thư viện lightgbm. Cài bằng: pip install lightgbm\n"
        "Hoặc: pip install -r requirements.txt"
    ) from exc


LIGHTGBM_PARAM_GRID = [
    {"n_estimators": 250, "learning_rate": 0.06, "num_leaves": 31, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 450, "learning_rate": 0.045, "num_leaves": 31, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 650, "learning_rate": 0.035, "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 450, "learning_rate": 0.045, "num_leaves": 63, "subsample": 0.85, "colsample_bytree": 0.85},
    {"n_estimators": 650, "learning_rate": 0.03, "num_leaves": 63, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 350, "learning_rate": 0.05, "num_leaves": 15, "subsample": 0.9, "colsample_bytree": 0.9},
]

RANDOM_FOREST_PARAM_GRID = [
    {"n_estimators": 180, "max_depth": 16, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 260, "max_depth": 20, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 320, "max_depth": 24, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 260, "max_depth": 20, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"n_estimators": 320, "max_depth": None, "min_samples_leaf": 3, "max_features": "log2"},
]

METRICS_TO_PLOT = ["f1", "roc_auc", "ndcg_at_10", "training_seconds"]


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )


def make_model(model_name: str, params: dict):
    if model_name == "lightgbm":
        return LGBMClassifier(
            **params,
            random_state=RANDOM_STATE,
            objective="binary",
            verbose=-1,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            **params,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        )
    raise ValueError(f"Model không hợp lệ: {model_name}")


def run_one_config(
    model_name: str,
    config_id: str,
    params: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[dict, Pipeline]:
    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    pipeline = Pipeline([
        ("preprocess", build_preprocessor(numeric_features, categorical_features)),
        ("model", make_model(model_name, params)),
    ])

    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_seconds = time.perf_counter() - started

    proba = pipeline.predict_proba(X_test)[:, 1]
    eval_df = test_df[["user_id", "song_id", TARGET]].copy()
    eval_df["pred"] = proba

    metrics = {
        "model": model_name,
        "config_id": config_id,
        "model_config": f"{model_name}_{config_id}",
        "params": params,
        "rows": int(len(train_df) + len(test_df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "training_seconds": round(float(training_seconds), 4),
        **classification_metrics(y_test, proba),
        **precision_recall_ndcg_at_k(eval_df, k=10),
    }
    return metrics, pipeline


def choose_best_by_metric(results: list[dict], metric: str) -> dict:
    return max(results, key=lambda row: row.get(metric, 0.0))


def save_best_single_model(results: list[dict], pipelines: dict[str, Pipeline], model_name: str, metric: str) -> None:
    model_results = [row for row in results if row["model"] == model_name]
    if not model_results:
        return
    best = choose_best_by_metric(model_results, metric)
    pipeline = pipelines[best["model_config"]]

    if model_name == "lightgbm":
        model_path = MODEL_DIR / "lightgbm_recommender.joblib"
        metrics_path = REPORT_DIR / "lightgbm_metrics.json"
        display_name = "LightGBMClassifier"
    else:
        model_path = MODEL_DIR / "random_forest_recommender.joblib"
        metrics_path = REPORT_DIR / "random_forest_metrics.json"
        display_name = "RandomForestClassifier"

    joblib.dump(pipeline, model_path)
    output_metrics = dict(best)
    output_metrics["model"] = display_name
    output_metrics["selected_by"] = metric
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, ensure_ascii=False, indent=2)


def save_results(results: list[dict]) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in results:
        flat = dict(row)
        flat["params_json"] = json.dumps(flat.pop("params"), ensure_ascii=False)
        rows.append(flat)

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "model_tuning_results.csv", index=False, encoding="utf-8-sig")
    with open(REPORT_DIR / "model_tuning_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return df


def plot_bar_metric(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_df = df.sort_values(["model", "config_id"]).copy()
    colors = plot_df["model"].map({"lightgbm": "#2563eb", "random_forest": "#16a34a"}).fillna("#64748b")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(plot_df["model_config"], plot_df[metric], color=colors)
    plt.title(f"So sánh {metric} theo từng bộ tham số")
    plt.xlabel("Bộ tham số")
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)

    best_idx = plot_df[metric].astype(float).idxmax()
    best_pos = list(plot_df.index).index(best_idx)
    bars[best_pos].set_color("#f97316")
    plt.text(best_pos, plot_df.loc[best_idx, metric], "BEST", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_precision_recall_at_10(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df.sort_values(["model", "config_id"]).copy()
    x = range(len(plot_df))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], plot_df["precision_at_10"], width=width, label="Precision@10", color="#2563eb")
    plt.bar([i + width / 2 for i in x], plot_df["recall_at_10"], width=width, label="Recall@10", color="#16a34a")
    plt.title("So sánh Precision@10 và Recall@10 theo từng bộ tham số")
    plt.xlabel("Bộ tham số")
    plt.ylabel("Giá trị")
    plt.xticks(list(x), plot_df["model_config"], rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_plots(df: pd.DataFrame) -> None:
    figure_dir = REPORT_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for metric in METRICS_TO_PLOT:
        plot_bar_metric(df, metric, figure_dir / f"tuning_{metric}.png")
    plot_precision_recall_at_10(df, figure_dir / "tuning_precision_recall_at_10.png")


def maybe_sample(df: pd.DataFrame, sample_rows: int | None) -> pd.DataFrame:
    if not sample_rows or sample_rows >= len(df):
        return df
    return df.sample(sample_rows, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "lightgbm", "random_forest"], default="all")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--save-best", action="store_true", help="Ghi đè model chính bằng config tốt nhất theo metric được chọn.")
    parser.add_argument("--best-metric", default="ndcg_at_10", choices=["f1", "roc_auc", "precision_at_10", "recall_at_10", "ndcg_at_10"])
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = maybe_sample(load_training_pairs_df(), args.sample_rows)
    df, numeric_features, categorical_features = prepare_features(raw_df)
    features = numeric_features + categorical_features
    train_df, test_df = safe_train_test_split(df)

    jobs: list[tuple[str, str, dict]] = []
    if args.model in {"all", "lightgbm"}:
        jobs.extend(("lightgbm", f"lgbm_{i:02d}", params) for i, params in enumerate(LIGHTGBM_PARAM_GRID, start=1))
    if args.model in {"all", "random_forest"}:
        jobs.extend(("random_forest", f"rf_{i:02d}", params) for i, params in enumerate(RANDOM_FOREST_PARAM_GRID, start=1))

    results = []
    pipelines: dict[str, Pipeline] = {}
    for model_name, config_id, params in jobs:
        print(f"Training {model_name} {config_id}: {params}")
        metrics, pipeline = run_one_config(
            model_name=model_name,
            config_id=config_id,
            params=params,
            train_df=train_df,
            test_df=test_df,
            features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        results.append(metrics)
        pipelines[metrics["model_config"]] = pipeline
        print(json.dumps(metrics, ensure_ascii=False, indent=2))

    result_df = save_results(results)
    create_plots(result_df)

    summary_cols = [
        "model_config",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "precision_at_10",
        "recall_at_10",
        "ndcg_at_10",
        "training_seconds",
    ]
    print("\nTuning summary:")
    print(result_df[summary_cols].sort_values(["model", "config_id"]).to_string(index=False))

    if args.save_best:
        if args.model in {"all", "lightgbm"}:
            save_best_single_model(results, pipelines, "lightgbm", args.best_metric)
        if args.model in {"all", "random_forest"}:
            save_best_single_model(results, pipelines, "random_forest", args.best_metric)
        print(f"Saved best model(s) by {args.best_metric}.")

    print("\nSaved:")
    print(REPORT_DIR / "model_tuning_results.csv")
    print(REPORT_DIR / "model_tuning_results.json")
    print(REPORT_DIR / "figures")


if __name__ == "__main__":
    main()
