from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR
from database import load_interactions_df, load_songs_df, load_users_df
from model.train_lightgbm import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# Tách chuỗi, xử lý giá trị thiếu
def safe_split_text(value: object, sep: str = "|") -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(sep) if item.strip()]

# Tính độ tương đồng Cosine giữa 2 vector
def cosine_score(user_values: np.ndarray, song_values: np.ndarray) -> float:
    user_norm = np.linalg.norm(user_values)
    song_norm = np.linalg.norm(song_values)
    if user_norm == 0 or song_norm == 0:
        return 0.0
    return float(np.clip(np.dot(user_values, song_values) / (user_norm * song_norm), 0, 1))

# Tính độ khớp thể loại giữa người dùng và bài hát.
def genre_match_score(user: pd.Series, song: pd.Series) -> float:
    user_genres = set(safe_split_text(user.get("favorite_genres", "")))
    user_detailed = set(safe_split_text(user.get("favorite_detailed_genres", "")))
    song_genres = {str(song.get("genre_top", "")).strip()}
    song_genres.update(safe_split_text(song.get("genres_titles_text", "")))
    song_genres.update(safe_split_text(song.get("genres_all_titles_text", "")))

    user_all = {g.lower() for g in user_genres | user_detailed if g}
    song_all = {g.lower() for g in song_genres if g}
    if not user_all or not song_all:
        return 0.0
    # Dùng chỉ số Jaccard (giao / hợp) để tính độ tương đồng tập hợp
    return len(user_all & song_all) / len(user_all | song_all)

# Tính điểm tương đồng ngôn ngữ.
def language_match_score(user: pd.Series, song: pd.Series) -> float:
    user_lang = str(user.get("language_preference", "unknown"))
    song_lang = str(song.get("language_code", "unknown"))
    if user_lang == "unknown":
        return 0.3
    return 1.0 if user_lang == song_lang else 0.0

# Chuẩn bị 2 vector đặc tính âm thanh (Audio Features) để so khớp Cosine.
def preference_similarity_score(user: pd.Series, song: pd.Series) -> float:
    # Vector sở thích của người dùng
    user_values = np.array([
        float(user["preferred_energy"]),
        float(user["preferred_valence"]),
        float(user["preferred_danceability"]),
        float(user["preferred_tempo"]),
        float(user.get("preferred_acousticness", 0.5)),
        float(user.get("preferred_instrumentalness", 0.5)),
        float(user.get("preferred_liveness", 0.5)),
        float(user.get("preferred_speechiness", 0.5)),
    ])
    # Vector đặc tính thực tế của bài hát
    song_values = np.array([
        float(song["energy"]),
        float(song["valence"]),
        float(song["danceability"]),
        float(song["tempo_norm"]),
        float(song.get("acousticness", 0.5)),
        float(song.get("instrumentalness", 0.5)),
        float(song.get("liveness", 0.5)),
        float(song.get("speechiness", 0.5)),
    ])
    return cosine_score(user_values, song_values)


# Xây dựng bộ đặc trưng cho các bài hát ứng viên
def build_candidate_features(user: pd.Series, songs: pd.DataFrame) -> pd.DataFrame:
    candidates = songs.copy()

    candidates["user_id"] = int(user["user_id"])
    candidates["age"] = int(user["age"])
    candidates["age_group"] = user["age_group"]
    candidates["gender"] = user["gender"]
    candidates["language_preference"] = user["language_preference"]
    candidates["favorite_genres"] = user.get("favorite_genres", "")
    candidates["favorite_detailed_genres"] = user.get("favorite_detailed_genres", "")
    candidates["preferred_energy"] = float(user["preferred_energy"])
    candidates["preferred_valence"] = float(user["preferred_valence"])
    candidates["preferred_danceability"] = float(user["preferred_danceability"])
    candidates["preferred_tempo"] = float(user["preferred_tempo"])
    candidates["preferred_acousticness"] = float(user.get("preferred_acousticness", 0.5))
    candidates["preferred_instrumentalness"] = float(user.get("preferred_instrumentalness", 0.5))
    candidates["preferred_liveness"] = float(user.get("preferred_liveness", 0.5))
    candidates["preferred_speechiness"] = float(user.get("preferred_speechiness", 0.5))

    candidates["preference_similarity"] = candidates.apply(lambda song: preference_similarity_score(user, song), axis=1)
    candidates["genre_match_score"] = candidates.apply(lambda song: genre_match_score(user, song), axis=1)
    candidates["language_match_score"] = candidates.apply(lambda song: language_match_score(user, song), axis=1)
    return candidates

# Hàm thực thi quy trình gợi ý chính cho người dùng.
def recommend(user_id: int, top_k: int = 10, genre: str | None = None) -> pd.DataFrame:
    users = load_users_df()
    songs = load_songs_df()
    interactions = load_interactions_df()
    model = joblib.load(MODEL_DIR / "lightgbm_recommender.joblib")

    # Tìm thông tin người dùng hiện tại
    user_rows = users[users["user_id"] == user_id]
    if user_rows.empty:
        raise ValueError(f"Không tìm thấy user_id={user_id}")
    user = user_rows.iloc[0]

    # Lọc bài hát: Loại bỏ những bài user đã nghe rồi
    seen_song_ids = set(interactions.loc[interactions["user_id"] == user_id, "song_id"].astype(int))
    candidates = songs[~songs["id"].astype(int).isin(seen_song_ids)].copy()
    if genre:
        candidates = candidates[candidates["genre_top"].str.lower() == genre.lower()].copy()
    
    # Xây dựng bộ đặc trưng cho các bài hát ứng viên (tính điểm Cosine, điểm khớp ngôn ngữ...)
    feature_df = build_candidate_features(user, candidates)
    features = [col for col in FEATURES if col in feature_df.columns]
    
    # Dùng AI để chấm điểm từng bài hát (predict_proba)
    feature_df["match_score"] = model.predict_proba(feature_df[features])[:, 1]
    
    # Sắp xếp theo điểm số từ cao xuống thấp và trả về Top K
    result_cols = [
        "id", "title", "artist_name", "album_title", "genre_top", "language_code",
        "duration", "energy", "valence", "danceability", "tempo_norm", "match_score",
    ]
    return feature_df.sort_values("match_score", ascending=False).head(top_k)[result_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--genre", type=str, default=None)
    args = parser.parse_args()

    users = load_users_df()
    user = users[users["user_id"] == args.user_id].iloc[0]
    print("User profile:")
    print(user.to_string())
    print("\nRecommendations:")
    recs = recommend(args.user_id, args.top_k, args.genre)
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
