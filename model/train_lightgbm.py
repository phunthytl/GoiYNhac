from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TARGET
from database import load_training_pairs_df

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit(
        "Thiếu thư viện lightgbm. Cài bằng: pip install lightgbm\n"
        "Hoặc: pip install -r requirements.txt"
    ) from exc


NUMERIC_FEATURES = [
    "age",
    "preferred_energy",
    "preferred_valence",
    "preferred_danceability",
    "preferred_tempo",
    "preferred_popularity",
    "preferred_acousticness",
    "preferred_instrumentalness",
    "preferred_liveness",
    "preferred_speechiness",
    "duration_norm",
    "energy",
    "valence",
    "danceability",
    "tempo_norm",
    "popularity",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "gender",
    "language_preference",
    "favorite_genres",
    "favorite_detailed_genres",
    "genre_top",
    "genres_titles_text",
    "genres_all_titles_text",
    "language_code",
]

REQUIRED_FEATURES = [
    "preferred_energy",
    "preferred_valence",
    "preferred_danceability",
    "preferred_tempo",
    "preferred_popularity",
    "preferred_acousticness",
    "preferred_instrumentalness",
    "preferred_liveness",
    "preferred_speechiness",
    "energy",
    "valence",
    "danceability",
    "tempo_norm",
    "popularity",
]


def precision_recall_ndcg_at_k(df: pd.DataFrame, k: int = 10) -> dict:
    precisions = []
    recalls = []
    ndcgs = []

    for _, group in df.groupby("user_id"):
        if len(group) < k:
            continue
        relevant_total = int((group[TARGET] == 1).sum())
        if relevant_total == 0:
            continue

        top = group.sort_values("pred", ascending=False).head(k)
        rel = (top[TARGET] == 1).astype(int).to_numpy()
        precisions.append(rel.sum() / k)
        recalls.append(rel.sum() / relevant_total)

        gains = rel / np.log2(np.arange(2, len(rel) + 2))
        dcg = gains.sum()
        ideal_rel = np.ones(min(relevant_total, k))
        idcg = (ideal_rel / np.log2(np.arange(2, len(ideal_rel) + 2))).sum()
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return {
        f"precision_at_{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall_at_{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg_at_{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "users_evaluated": int(len(precisions)),
    }


def classification_metrics(y_true: pd.Series, proba: np.ndarray) -> dict:
    pred_label = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred_label)),
        "precision": float(precision_score(y_true, pred_label, zero_division=0)),
        "recall": float(recall_score(y_true, pred_label, zero_division=0)),
        "f1": float(f1_score(y_true, pred_label, zero_division=0)),
    }
    if len(set(y_true.astype(int))) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def userwise_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts = []
    test_parts = []

    for _, group in df.groupby("user_id", sort=False):
        user_train_parts = []
        user_test_parts = []

        for label in [0, 1]:
            label_group = group[group[TARGET] == label]
            if label_group.empty:
                continue
            indices = label_group.index.to_numpy()
            rng.shuffle(indices)

            if len(indices) >= 2:
                n_test = max(1, int(round(len(indices) * test_size)))
                n_test = min(n_test, len(indices) - 1)
                test_idx = indices[:n_test]
                train_idx = indices[n_test:]
                user_test_parts.append(df.loc[test_idx])
                user_train_parts.append(df.loc[train_idx])
            else:
                user_train_parts.append(label_group)

        if user_train_parts:
            train_parts.append(pd.concat(user_train_parts, ignore_index=False))
        if user_test_parts:
            test_parts.append(pd.concat(user_test_parts, ignore_index=False))

    if not train_parts or not test_parts:
        labels = df[TARGET].astype(int)
        value_counts = labels.value_counts()
        stratify = labels if len(value_counts) >= 2 and value_counts.min() >= 2 else None
        return train_test_split(df, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify)

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return train_df, test_df


safe_train_test_split = userwise_train_test_split


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    if df.empty:
        raise ValueError("Bảng training_pairs đang rỗng. Hãy chạy generate_synthetic_data.py trước.")
    if TARGET not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {TARGET}")

    missing_core = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_core:
        raise ValueError(
            "training_pairs thiếu cột quan trọng: "
            + ", ".join(missing_core)
            + "\nHãy chạy lại preprocess_songs.py và generate_synthetic_data.py."
        )

    out = df.dropna(subset=[TARGET]).copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1)
    numeric_features = [col for col in NUMERIC_FEATURES if col in out.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in out.columns]

    if not numeric_features and not categorical_features:
        raise ValueError("Không có feature hợp lệ để train model.")

    for col in numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median()).fillna(0.0)

    for col in categorical_features:
        out[col] = out[col].fillna("unknown").astype(str)

    return out, numeric_features, categorical_features


def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df, numeric_features, categorical_features = prepare_features(load_training_pairs_df())
    features = numeric_features + categorical_features

    train_df, test_df = safe_train_test_split(df)
    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test = test_df[TARGET]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = LGBMClassifier(
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        objective="binary",
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    eval_df = test_df[["user_id", "song_id", TARGET]].copy()
    eval_df["pred"] = proba
    ranking = precision_recall_ndcg_at_k(eval_df, k=10)

    metrics = {
        "model": "LightGBMClassifier",
        "target": TARGET,
        "rows": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        **classification_metrics(y_test, proba),
        **ranking,
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
    }

    joblib.dump(pipeline, MODEL_DIR / "lightgbm_recommender.joblib")
    with open(REPORT_DIR / "lightgbm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
