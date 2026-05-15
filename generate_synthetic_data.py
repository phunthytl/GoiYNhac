import argparse
import random

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    DEFAULT_INTERACTIONS_PER_USER,
    DEFAULT_USER_COUNT,
    RANDOM_STATE,
)

from database import (
    import_synthetic_users_with_accounts,
    import_training_data,
    load_songs_df,
)


GENDERS = ["Male", "Female", "Other"]
LANG_PREFS = ["en", "es", "fr", "de", "unknown"]
DEMO_PASSWORD = "User@123456"


def age_group(age: int) -> str:
    if age < 20:
        return "teen"
    if age < 30:
        return "young_adult"
    if age < 45:
        return "adult"
    return "mature"


def safe_split_text(value: object, sep: str = "|") -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(sep) if item.strip()]


def cosine_score(user_values: np.ndarray, song_values: np.ndarray) -> float:
    user_norm = np.linalg.norm(user_values)
    song_norm = np.linalg.norm(song_values)
    if user_norm == 0 or song_norm == 0:
        return 0.0
    score = float(np.dot(user_values, song_values) / (user_norm * song_norm))
    return float(np.clip(score, 0, 1))


def validate_songs_columns(songs: pd.DataFrame) -> None:
    required_cols = [
        "id",
        "genre_top",
        "language_code",
        "duration",
        "energy",
        "valence",
        "danceability",
        "tempo_norm",
        "popularity",
    ]
    missing_cols = [col for col in required_cols if col not in songs.columns]
    if missing_cols:
        raise ValueError(
            "Bảng songs đang thiếu các cột bắt buộc: "
            + ", ".join(missing_cols)
            + "\nHãy chạy lại preprocess_songs.py trước."
        )


def make_users(songs: pd.DataFrame, n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    genres = songs["genre_top"].dropna().unique().tolist()
    if not genres:
        raise ValueError("Không có genre_top hợp lệ trong bảng songs.")

    genre_counts = songs["genre_top"].value_counts(normalize=True)
    genre_probs = genre_counts.reindex(genres).fillna(0).to_numpy()
    genre_probs = genre_probs / genre_probs.sum()
    rows = []

    for user_id in range(1, n_users + 1):
        age = int(rng.integers(16, 61))
        gender = str(rng.choice(GENDERS, p=[0.34, 0.34, 0.32]))
        max_genres = min(5, len(genres))
        k = int(rng.integers(2, max_genres + 1))
        fav_genres = rng.choice(genres, size=k, replace=False, p=genre_probs).tolist()

        favorite_detailed_genres = ""
        if "genres_all_titles_text" in songs.columns:
            detailed_pool = songs.loc[
                songs["genre_top"].isin(fav_genres),
                "genres_all_titles_text",
            ].fillna("")
            if len(detailed_pool) > 0:
                sampled_texts = detailed_pool.sample(
                    min(len(detailed_pool), 80),
                    random_state=int(rng.integers(0, 1_000_000)),
                )
                detailed = sorted({genre for text in sampled_texts for genre in safe_split_text(text)})
                if detailed:
                    selected_count = min(len(detailed), int(rng.integers(1, 4)))
                    favorite_detailed_genres = "|".join(
                        rng.choice(detailed, size=selected_count, replace=False).tolist()
                    )

        rows.append(
            {
                "user_id": user_id,
                "age": age,
                "age_group": age_group(age),
                "gender": gender,
                "favorite_genres": "|".join(fav_genres),
                "language_preference": str(rng.choice(LANG_PREFS, p=[0.55, 0.08, 0.08, 0.04, 0.25])),
                "preferred_energy": float(rng.beta(2.2, 2.2)),
                "preferred_valence": float(rng.beta(2.0, 2.0)),
                "preferred_danceability": float(rng.beta(2.0, 2.3)),
                "preferred_tempo": float(rng.beta(2.2, 2.1)),
                "preferred_popularity": float(rng.beta(1.8, 2.8)),
                "preferred_acousticness": float(rng.beta(2.0, 2.2)),
                "preferred_instrumentalness": float(rng.beta(1.5, 3.0)),
                "preferred_liveness": float(rng.beta(1.6, 3.2)),
                "preferred_speechiness": float(rng.beta(1.4, 3.5)),
                "favorite_detailed_genres": favorite_detailed_genres,
            }
        )

    return pd.DataFrame(rows)


def genre_match_score(user: pd.Series, song: pd.Series) -> float:
    user_genres = set(safe_split_text(user.get("favorite_genres", "")))
    user_detailed = set(safe_split_text(user.get("favorite_detailed_genres", "")))
    song_genres = {str(song.get("genre_top", "")).strip()}
    if "genres_titles_text" in song:
        song_genres.update(safe_split_text(song.get("genres_titles_text", "")))
    if "genres_all_titles_text" in song:
        song_genres.update(safe_split_text(song.get("genres_all_titles_text", "")))

    user_all = {genre.lower() for genre in user_genres | user_detailed if genre}
    song_all = {genre.lower() for genre in song_genres if genre}
    if not user_all or not song_all:
        return 0.0
    return len(user_all & song_all) / len(user_all | song_all)


def language_match_score(user: pd.Series, song: pd.Series) -> float:
    user_lang = str(user.get("language_preference", "unknown"))
    song_lang = str(song.get("language_code", "unknown"))
    if user_lang == "unknown":
        return 0.3
    return 1.0 if user_lang == song_lang else 0.0


def preference_similarity_score(user: pd.Series, song: pd.Series) -> float:
    user_values = np.array(
        [
            float(user["preferred_energy"]),
            float(user["preferred_valence"]),
            float(user["preferred_danceability"]),
            float(user["preferred_tempo"]),
            float(user["preferred_popularity"]),
            float(user.get("preferred_acousticness", 0.5)),
            float(user.get("preferred_instrumentalness", 0.5)),
            float(user.get("preferred_liveness", 0.5)),
            float(user.get("preferred_speechiness", 0.5)),
        ],
        dtype=float,
    )
    song_values = np.array(
        [
            float(song["energy"]),
            float(song["valence"]),
            float(song["danceability"]),
            float(song["tempo_norm"]),
            float(song["popularity"]),
            float(song.get("acousticness", 0.5)),
            float(song.get("instrumentalness", 0.5)),
            float(song.get("liveness", 0.5)),
            float(song.get("speechiness", 0.5)),
        ],
        dtype=float,
    )
    return cosine_score(user_values, song_values)


def listen_affinity_score(user: pd.Series, song: pd.Series, rng: np.random.Generator) -> float:
    score = (
        0.65 * preference_similarity_score(user, song)
        + 0.25 * genre_match_score(user, song)
        + 0.10 * language_match_score(user, song)
        + rng.normal(0, 0.08)
    )
    return float(score)


def make_target_counts(
    n_users: int,
    total_interactions: int,
    rng: np.random.Generator,
    max_items_per_user: int,
) -> np.ndarray:
    if total_interactions < n_users:
        raise ValueError("total_interactions phải >= n_users để mỗi user có ít nhất 1 bài đã nghe.")
    if total_interactions > n_users * max_items_per_user:
        raise ValueError("total_interactions vượt quá số cặp user-song tối đa có thể sinh.")

    weights = rng.gamma(shape=2.2, scale=1.0, size=n_users)
    counts = np.floor(weights / weights.sum() * total_interactions).astype(int)
    counts = np.maximum(counts, 1)
    counts = np.minimum(counts, max_items_per_user)

    while counts.sum() < total_interactions:
        candidates = np.where(counts < max_items_per_user)[0]
        add_count = min(total_interactions - int(counts.sum()), len(candidates))
        selected = rng.choice(candidates, size=add_count, replace=False)
        counts[selected] += 1

    while counts.sum() > total_interactions:
        candidates = np.where(counts > 1)[0]
        remove_count = min(int(counts.sum()) - total_interactions, len(candidates))
        selected = rng.choice(candidates, size=remove_count, replace=False)
        counts[selected] -= 1

    return counts


def sample_interaction_count(rng: np.random.Generator) -> int:
    bucket = rng.choice(["very_light", "light", "medium", "active", "power"], p=[0.10, 0.50, 0.25, 0.12, 0.03])
    if bucket == "very_light":
        return int(rng.integers(3, 8))
    if bucket == "light":
        return int(rng.integers(15, 26))
    if bucket == "medium":
        return int(rng.integers(30, 46))
    if bucket == "active":
        return int(rng.integers(50, 76))
    return int(rng.integers(90, 131))


def build_training_pair(user: pd.Series, song: pd.Series, listened: int) -> dict:
    return {
        "user_id": int(user["user_id"]),
        "song_id": int(song["id"]),
        "listened": int(listened),
        "age": int(user["age"]),
        "age_group": user["age_group"],
        "gender": user["gender"],
        "language_preference": user["language_preference"],
        "favorite_genres": user["favorite_genres"],
        "favorite_detailed_genres": user["favorite_detailed_genres"],
        "preferred_energy": float(user["preferred_energy"]),
        "preferred_valence": float(user["preferred_valence"]),
        "preferred_danceability": float(user["preferred_danceability"]),
        "preferred_tempo": float(user["preferred_tempo"]),
        "preferred_popularity": float(user["preferred_popularity"]),
        "preferred_acousticness": float(user.get("preferred_acousticness", 0.5)),
        "preferred_instrumentalness": float(user.get("preferred_instrumentalness", 0.5)),
        "preferred_liveness": float(user.get("preferred_liveness", 0.5)),
        "preferred_speechiness": float(user.get("preferred_speechiness", 0.5)),
        "genre_top": song["genre_top"],
        "genres_titles_text": song.get("genres_titles_text", ""),
        "genres_all_titles_text": song.get("genres_all_titles_text", ""),
        "language_code": song["language_code"],
        "duration": float(song.get("duration", 0)),
        "duration_norm": float(song.get("duration_norm", 0.5)),
        "energy": float(song["energy"]),
        "valence": float(song["valence"]),
        "danceability": float(song["danceability"]),
        "tempo_norm": float(song["tempo_norm"]),
        "popularity": float(song["popularity"]),
        "acousticness": float(song.get("acousticness", 0.5)),
        "instrumentalness": float(song.get("instrumentalness", 0.5)),
        "liveness": float(song.get("liveness", 0.5)),
        "speechiness": float(song.get("speechiness", 0.5)),
        "preference_similarity": preference_similarity_score(user, song),
        "genre_match_score": genre_match_score(user, song),
        "language_match_score": language_match_score(user, song),
    }


def sample_positive_songs(
    user: pd.Series,
    songs: pd.DataFrame,
    by_genre: dict[str, pd.DataFrame],
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    fav_genres = safe_split_text(user["favorite_genres"])
    candidate_parts = []
    if fav_genres:
        for genre in fav_genres:
            group = by_genre.get(genre)
            if group is not None and not group.empty:
                candidate_parts.append(group)
    candidate_parts.append(songs.sample(min(len(songs), target_n * 4), random_state=int(rng.integers(0, 1_000_000))))
    candidates = pd.concat(candidate_parts, ignore_index=True).drop_duplicates(subset=["id"])
    if len(candidates) < target_n:
        candidates = songs.copy()

    scored = candidates.copy()
    scored["_affinity"] = scored.apply(lambda song: listen_affinity_score(user, song, rng), axis=1)
    pool_size = min(len(scored), max(target_n * 3, target_n))
    pool = scored.nlargest(pool_size, "_affinity")
    return pool.sample(target_n, replace=False, random_state=int(rng.integers(0, 1_000_000))).drop(columns=["_affinity"])


def generate(
    n_users: int,
    interactions_per_user: int | None = None,
    total_interactions: int | None = None,
    negative_ratio: float = 1.0,
    random_interactions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    songs = load_songs_df()
    validate_songs_columns(songs)
    rng = np.random.default_rng(RANDOM_STATE)
    users = make_users(songs, n_users, rng)
    import_synthetic_users_with_accounts(users, password=DEMO_PASSWORD)

    by_genre = {genre: group for genre, group in songs.groupby("genre_top")}
    target_counts = None
    if total_interactions is not None:
        target_counts = make_target_counts(n_users, total_interactions, rng, len(songs))

    interactions_rows = []
    training_pair_rows = []

    for user_index, user in enumerate(users.itertuples(index=False)):
        user_s = pd.Series(user._asdict())
        if target_counts is not None:
            target_n = int(target_counts[user_index])
        else:
            target_n = sample_interaction_count(rng) if random_interactions else int(interactions_per_user)

        positive_songs = sample_positive_songs(user_s, songs, by_genre, target_n, rng)
        positive_ids = set(positive_songs["id"].astype(int).tolist())

        for _, song in positive_songs.iterrows():
            song_id = int(song["id"])
            interactions_rows.append({"user_id": int(user_s["user_id"]), "song_id": song_id, "listened": 1})
            training_pair_rows.append(build_training_pair(user_s, song, listened=1))

        negative_n = int(round(target_n * negative_ratio))
        negative_pool = songs[~songs["id"].astype(int).isin(positive_ids)]
        if negative_n > 0 and not negative_pool.empty:
            negative_n = min(negative_n, len(negative_pool))
            negative_songs = negative_pool.sample(negative_n, replace=False, random_state=int(rng.integers(0, 1_000_000)))
            for _, song in negative_songs.iterrows():
                training_pair_rows.append(build_training_pair(user_s, song, listened=0))

    interactions = pd.DataFrame(interactions_rows)
    training_pairs = pd.DataFrame(training_pair_rows)
    import_training_data(interactions, training_pairs)
    return users, interactions, training_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=DEFAULT_USER_COUNT)
    parser.add_argument("--interactions-per-user", type=int, default=DEFAULT_INTERACTIONS_PER_USER)
    parser.add_argument("--fixed-interactions", action="store_true", help="Dùng cùng một số bài đã nghe cho mọi user.")
    parser.add_argument("--total-interactions", type=int, default=None, help="Tổng số bài đã nghe cần sinh chính xác.")
    parser.add_argument("--negative-ratio", type=float, default=1.0, help="Số negative samples trên mỗi positive sample trong training_pairs.")
    args = parser.parse_args()

    users, interactions, pairs = generate(
        n_users=args.users,
        interactions_per_user=args.interactions_per_user,
        total_interactions=args.total_interactions,
        negative_ratio=args.negative_ratio,
        random_interactions=not args.fixed_interactions,
    )

    print(f"Saved accounts/users: {len(users)}")
    print(f"Demo login: user0001 / {DEMO_PASSWORD}")
    print(f"Saved listened interactions: {len(interactions)}")
    print(f"Saved training pairs: {len(pairs)}")
    print("Interactions/user distribution:")
    print(interactions.groupby("user_id").size().describe().to_string())
    print("Training label distribution:")
    print(pairs["listened"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
