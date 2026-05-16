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


# Danh sách các đặc trưng dạng số
NUMERIC_FEATURES = [
    "preference_similarity",
    "genre_match_score",
    "language_match_score",
    "age",
    "preferred_energy",
    "preferred_valence",
    "preferred_danceability",
    "preferred_tempo",
    "preferred_acousticness",
    "preferred_instrumentalness",
    "preferred_liveness",
    "preferred_speechiness",
    "duration_norm",       
    "energy",             
    "valence",                 
    "danceability",           
    "tempo_norm",            
    "acousticness",           
    "instrumentalness", 
    "liveness",                
    "speechiness",             
]

# Danh sách các đặc trưng dạng chữ
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

# Kiểm tra xem dữ liệu có đủ các cột bắt buộc không
REQUIRED_FEATURES = [
    "preferred_energy",
    "preferred_valence",
    "preferred_danceability",
    "preferred_tempo",
    "preferred_acousticness",
    "preferred_instrumentalness",
    "preferred_liveness",
    "preferred_speechiness",
    "energy",
    "valence",
    "danceability",
    "tempo_norm",
]

# Tính toán độ chính xác và độ phủ ở top k
def precision_recall_ndcg_at_k(df: pd.DataFrame, k: int = 10) -> dict:
    precisions = []
    recalls = []
    ndcgs = []

    # Lấy từng nhóm dữ liệu của từng người dùng
    for _, group in df.groupby("user_id"):
        if len(group) < k:
            continue
        relevant_total = int((group[TARGET] == 1).sum())
        if relevant_total == 0:
            continue

        # Lấy top k bài hát được đề xuất
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


# Tính toán các chỉ số đánh giá khác
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

# Xử lý dữ liệu (Loại bỏ NaN, ép kiểu, xử lý text, chia train/test)
def userwise_train_test_split(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts = []
    test_parts = []

    for _, group in df.groupby("user_id", sort=False):
        user_train_parts = []
        user_test_parts = []
        # Lấy dữ liệu của từng người dùng
        for label in [0, 1]:
            label_group = group[group[TARGET] == label]
            if label_group.empty:
                continue
            indices = label_group.index.to_numpy()
            rng.shuffle(indices)

            # Chia dữ liệu của từng người dùng thành tập train và test
            if len(indices) >= 2:
                n_test = max(1, int(round(len(indices) * test_size)))
                n_test = min(n_test, len(indices) - 1)
                test_idx = indices[:n_test]
                train_idx = indices[n_test:]
                user_test_parts.append(df.loc[test_idx])
                user_train_parts.append(df.loc[train_idx])
            else:
                user_train_parts.append(label_group)

        # Ghép lại dữ liệu của từng người dùng
        if user_train_parts:
            train_parts.append(pd.concat(user_train_parts, ignore_index=False))
        if user_test_parts:
            test_parts.append(pd.concat(user_test_parts, ignore_index=False))

    # Nếu không có dữ liệu cho train hoặc test, sử dụng train_test_split thông thường
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
    """
    Hàm tiền xử lý dữ liệu trước khi đưa vào huấn luyện.
    Nhiệm vụ: Kiểm tra cột, xử lý giá trị thiếu (NaN), ép kiểu dữ liệu.
    """
    if df.empty:
        raise ValueError("Bảng training_pairs đang rỗng. Hãy chạy generate_synthetic_data.py trước.")
    
    # Loại bỏ các dòng không có nhãn (target) và ép kiểu target về 0 hoặc 1
    out = df.dropna(subset=[TARGET]).copy()
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1)

    # Lọc lấy danh sách các đặc trưng thực tế có trong dataframe
    numeric_features = [col for col in NUMERIC_FEATURES if col in out.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in out.columns]

    # Xử lý đặc trưng số: Điền giá trị trung vị (median) vào chỗ trống để tránh lỗi AI
    for col in numeric_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].fillna(out[col].median()).fillna(0.0)

    # Xử lý đặc trưng chữ: Điền "unknown" vào chỗ trống và ép kiểu string
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

    # 1. Công đoạn tiền xử lý (Preprocess): Biến đổi dữ liệu thô sang dạng máy hiểu được
    preprocessor = ColumnTransformer(
        transformers=[
            # Cột số: Giữ nguyên (passthrough)
            ("num", "passthrough", numeric_features),
            # Cột chữ: Dùng OneHotEncoder để biến thành các vector 0, 1
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop", # Bỏ qua các cột không khai báo
    )

    # 2. Cấu hình mô hình LightGBM
    model = LGBMClassifier(
        n_estimators=450,    # Số lượng cây quyết định
        learning_rate=0.045, # Tốc độ học
        num_leaves=31,       # Số lá trên mỗi cây
        subsample=0.85,      # Tỉ lệ lấy mẫu dữ liệu
        colsample_bytree=0.85, # Tỉ lệ lấy mẫu đặc trưng
        random_state=RANDOM_STATE,
        objective="binary",  # Phân loại 0 hoặc 1
    )

    # 3. Ghép nối 2 công đoạn thành 1 Pipeline hoàn chỉnh
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
