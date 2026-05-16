import argparse
import random
from datetime import datetime

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
UNKNOWN_LANGUAGE_PROB = 0.08
GENRE_EXPLORATION_PROB = 0.45
LANGUAGE_EXPLORATION_PROB = 0.50
AFFINITY_NOISE_STD = 0.12
HARD_NEGATIVE_RATIO = 0.75
HARD_NEGATIVE_POOL_MULTIPLIER = 5
HARD_NEGATIVE_SCORE_POOL_MULTIPLIER = 30
HARD_NEGATIVE_MIN_SCORE_POOL = 500
DEMO_PASSWORD = "User@123456"


def log_step(message: str) -> None:
    """In tiến trình ra terminal ngay lập tức để biết pipeline đang chạy tới đâu."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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

    language_counts = songs["language_code"].fillna("unknown").astype(str).value_counts()
    language_counts = language_counts[language_counts.index != "unknown"]
    language_codes = language_counts.index.tolist()
    language_probs = language_counts.to_numpy(dtype=float)
    if language_codes:
        language_probs = language_probs / language_probs.sum()

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

        if language_codes and rng.random() > UNKNOWN_LANGUAGE_PROB:
            language_preference = str(rng.choice(language_codes, p=language_probs))
        else:
            language_preference = "unknown"

        rows.append(
            {
                "user_id": user_id,
                "age": age,
                "age_group": age_group(age),
                "gender": gender,
                "favorite_genres": "|".join(fav_genres),
                "language_preference": language_preference,
                "preferred_energy": float(rng.beta(2.2, 2.2)),
                "preferred_valence": float(rng.beta(2.0, 2.0)),
                "preferred_danceability": float(rng.beta(2.0, 2.3)),
                "preferred_tempo": float(rng.beta(2.2, 2.1)),
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
        0.40 * preference_similarity_score(user, song)
        + 0.25 * genre_match_score(user, song)
        + 0.15 * language_match_score(user, song)
        + rng.normal(0, AFFINITY_NOISE_STD)
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
        "acousticness": float(song.get("acousticness", 0.5)),
        "instrumentalness": float(song.get("instrumentalness", 0.5)),
        "liveness": float(song.get("liveness", 0.5)),
        "speechiness": float(song.get("speechiness", 0.5)),
        "preference_similarity": preference_similarity_score(user, song),
        "genre_match_score": genre_match_score(user, song),
        "language_match_score": language_match_score(user, song),
    }


def score_and_sample_positive_pool(
    user: pd.Series,
    candidates: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    scored = candidates.copy()
    scored["_affinity"] = scored.apply(lambda song: listen_affinity_score(user, song, rng), axis=1)
    pool_size = min(len(scored), max(target_n * 3, target_n))
    pool = scored.nlargest(pool_size, "_affinity")
    sample_n = min(target_n, len(pool))
    return pool.sample(sample_n, replace=False, random_state=int(rng.integers(0, 1_000_000))).drop(columns=["_affinity"])


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

    genre_candidates = pd.concat(candidate_parts, ignore_index=True).drop_duplicates(subset=["id"]) if candidate_parts else pd.DataFrame()
    language_preference = str(user.get("language_preference", "unknown"))

    in_genre_n = int(round(target_n * (1 - GENRE_EXPLORATION_PROB)))
    exploration_n = target_n - in_genre_n
    selected_parts = []
    selected_ids: set[int] = set()

    if not genre_candidates.empty and in_genre_n > 0:
        primary_candidates = genre_candidates
        if language_preference != "unknown":
            language_target_n = int(round(in_genre_n * (1 - LANGUAGE_EXPLORATION_PROB)))
            language_candidates = genre_candidates[genre_candidates["language_code"].astype(str) == language_preference].copy()
            if language_target_n > 0 and not language_candidates.empty:
                language_primary = score_and_sample_positive_pool(user, language_candidates, language_target_n, rng)
                selected_parts.append(language_primary)
                selected_ids.update(language_primary["id"].astype(int).tolist())

            remaining_primary_n = in_genre_n - len(selected_ids)
            primary_pool = genre_candidates[~genre_candidates["id"].astype(int).isin(selected_ids)].copy()
            primary = score_and_sample_positive_pool(user, primary_pool, remaining_primary_n, rng)
            selected_parts.append(primary)
            selected_ids.update(primary["id"].astype(int).tolist())
            primary = pd.DataFrame()
        else:
            primary = score_and_sample_positive_pool(user, primary_candidates, in_genre_n, rng)
            selected_parts.append(primary)
            selected_ids.update(primary["id"].astype(int).tolist())

    remaining_n = target_n - len(selected_ids)
    if remaining_n > 0:
        exploration_pool = songs[~songs["id"].astype(int).isin(selected_ids)].copy()
        if exploration_n <= 0 and not genre_candidates.empty:
            exploration_pool = genre_candidates[~genre_candidates["id"].astype(int).isin(selected_ids)].copy()
        exploration = score_and_sample_positive_pool(user, exploration_pool, remaining_n, rng)
        selected_parts.append(exploration)
        selected_ids.update(exploration["id"].astype(int).tolist())

    positives = pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["id"]) if selected_parts else pd.DataFrame()
    if len(positives) < target_n:
        fill_pool = songs[~songs["id"].astype(int).isin(positives["id"].astype(int).tolist())].copy()
        fill = score_and_sample_positive_pool(user, fill_pool, target_n - len(positives), rng)
        positives = pd.concat([positives, fill], ignore_index=True).drop_duplicates(subset=["id"])

    return positives.head(target_n)


def negative_hardness_score(user: pd.Series, song: pd.Series, rng: np.random.Generator) -> float:
    """
    Tính độ "khó" của một negative sample.

    Negative khó là bài hát nhìn qua vẫn có vẻ hợp gu user: có thể cùng genre,
    cùng ngôn ngữ hoặc audio features tương đối gần, nhưng vẫn được gán nhãn
    listened=0. Cách này làm positive/negative giống nhau hơn, buộc model học
    tinh hơn thay vì chỉ dựa vào vài rule quá rõ.
    """
    score = (
        0.45 * preference_similarity_score(user, song)
        + 0.35 * genre_match_score(user, song)
        + 0.20 * language_match_score(user, song)
        + rng.normal(0, AFFINITY_NOISE_STD / 2)
    )
    return float(score)


def sample_negative_songs(
    user: pd.Series,
    songs: pd.DataFrame,
    positive_ids: set[int],
    target_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Sinh negative samples theo hướng khó học hơn.

    Trước đây negative được sample ngẫu nhiên từ toàn bộ bài chưa nghe. Cách đó
    thường tạo nhiều negative quá khác gu user, khiến model phân biệt rất dễ và
    LightGBM hay cho xác suất sát 1.0.

    Hàm này chọn phần lớn negative từ nhóm bài có điểm gần gu user nhất
    (hard negatives), sau đó trộn thêm một phần random negatives để dữ liệu vẫn
    đa dạng.
    """
    if target_n <= 0:
        return pd.DataFrame()

    candidate_pool = songs[~songs["id"].astype(int).isin(positive_ids)].copy()
    if candidate_pool.empty:
        return candidate_pool

    target_n = min(target_n, len(candidate_pool))
    hard_n = int(round(target_n * HARD_NEGATIVE_RATIO))
    random_n = target_n - hard_n

    selected_parts = []
    selected_ids: set[int] = set()

    if hard_n > 0:
        fav_genres = set(safe_split_text(user.get("favorite_genres", "")))
        language_preference = str(user.get("language_preference", "unknown"))

        priority_mask = candidate_pool["genre_top"].isin(fav_genres)
        if language_preference != "unknown":
            priority_mask = priority_mask | (candidate_pool["language_code"].astype(str) == language_preference)

        priority_pool = candidate_pool[priority_mask].copy()
        if priority_pool.empty:
            priority_pool = candidate_pool

        score_pool_size = min(
            len(priority_pool),
            max(hard_n * HARD_NEGATIVE_SCORE_POOL_MULTIPLIER, HARD_NEGATIVE_MIN_SCORE_POOL),
        )
        scored = priority_pool.sample(
            score_pool_size,
            replace=False,
            random_state=int(rng.integers(0, 1_000_000)),
        ).copy()
        scored["_hardness"] = scored.apply(lambda song: negative_hardness_score(user, song, rng), axis=1)
        pool_size = min(len(scored), max(hard_n * HARD_NEGATIVE_POOL_MULTIPLIER, hard_n))
        hard_pool = scored.nlargest(pool_size, "_hardness")
        hard_samples = hard_pool.sample(
            min(hard_n, len(hard_pool)),
            replace=False,
            random_state=int(rng.integers(0, 1_000_000)),
        ).drop(columns=["_hardness"])
        selected_parts.append(hard_samples)
        selected_ids.update(hard_samples["id"].astype(int).tolist())

    remaining_pool = candidate_pool[~candidate_pool["id"].astype(int).isin(selected_ids)]
    if random_n > 0 and not remaining_pool.empty:
        random_samples = remaining_pool.sample(
            min(random_n, len(remaining_pool)),
            replace=False,
            random_state=int(rng.integers(0, 1_000_000)),
        )
        selected_parts.append(random_samples)
        selected_ids.update(random_samples["id"].astype(int).tolist())

    if len(selected_ids) < target_n:
        fill_pool = candidate_pool[~candidate_pool["id"].astype(int).isin(selected_ids)]
        if not fill_pool.empty:
            fill = fill_pool.sample(
                min(target_n - len(selected_ids), len(fill_pool)),
                replace=False,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            selected_parts.append(fill)

    if not selected_parts:
        return pd.DataFrame()
    return pd.concat(selected_parts, ignore_index=True).drop_duplicates(subset=["id"]).head(target_n)


def generate(
    n_users: int,
    interactions_per_user: int | None = None,
    total_interactions: int | None = None,
    negative_ratio: float = 1.0,
    random_interactions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log_step("Bắt đầu sinh dữ liệu mô phỏng.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log_step("Đang đọc bảng songs từ SQLite.")
    songs = load_songs_df()
    log_step(f"Đã đọc {len(songs)} bài hát.")

    log_step("Đang kiểm tra các cột bắt buộc của bảng songs.")
    validate_songs_columns(songs)

    rng = np.random.default_rng(RANDOM_STATE)

    log_step(f"Đang sinh {n_users} hồ sơ người dùng mô phỏng.")
    users = make_users(songs, n_users, rng)
    log_step("Đã sinh hồ sơ người dùng.")

    log_step("Đang lưu accounts demo và user_profiles vào database.")
    import_synthetic_users_with_accounts(users, password=DEMO_PASSWORD)
    log_step("Đã lưu accounts demo và user_profiles.")

    log_step("Đang nhóm bài hát theo genre để tăng tốc sampling.")
    by_genre = {genre: group for genre, group in songs.groupby("genre_top")}

    target_counts = None
    if total_interactions is not None:
        log_step(f"Đang phân bổ chính xác {total_interactions} interactions cho {n_users} users.")
        target_counts = make_target_counts(n_users, total_interactions, rng, len(songs))
        log_step("Đã phân bổ số interactions cho từng user.")

    interactions_rows = []
    training_pair_rows = []

    log_step("Đang sinh interactions và training_pairs cho từng user.")
    progress_every = max(1, n_users // 20)
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
        negative_songs = sample_negative_songs(user_s, songs, positive_ids, negative_n, rng)
        if not negative_songs.empty:
            for _, song in negative_songs.iterrows():
                training_pair_rows.append(build_training_pair(user_s, song, listened=0))

        processed = user_index + 1
        if processed == 1 or processed == n_users or processed % progress_every == 0:
            log_step(
                "Tiến trình: "
                f"{processed}/{n_users} users | "
                f"interactions={len(interactions_rows)} | "
                f"training_pairs={len(training_pair_rows)}"
            )

    log_step("Đang chuyển dữ liệu đã sinh thành DataFrame.")
    interactions = pd.DataFrame(interactions_rows)
    training_pairs = pd.DataFrame(training_pair_rows)

    log_step(
        "Đang lưu interactions và training_pairs vào database "
        f"({len(interactions)} interactions, {len(training_pairs)} training pairs)."
    )
    import_training_data(interactions, training_pairs)
    log_step("Hoàn tất sinh dữ liệu mô phỏng.")
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
