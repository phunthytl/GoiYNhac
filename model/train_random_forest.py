from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TARGET
from database import load_training_pairs_df
from model.train_lightgbm import classification_metrics, prepare_features, precision_recall_ndcg_at_k, safe_train_test_split


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

    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
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
        "model": "RandomForestClassifier",
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
        "params": {
            "n_estimators": 260,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": RANDOM_STATE,
        },
    }

    joblib.dump(pipeline, MODEL_DIR / "random_forest_recommender.joblib")
    with open(REPORT_DIR / "random_forest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    train()
