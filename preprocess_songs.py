import argparse
import ast

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DATA_DIR, RAW_DIR, DEFAULT_SONG_LIMIT


def parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_genre_map() -> dict[int, str]:
    genres = pd.read_csv(RAW_DIR / "genres.csv")
    return dict(
        zip(
            genres["genre_id"].astype(int),
            genres["title"].astype(str),
        )
    )


def genre_ids_to_titles(values, genre_map: dict[int, str]) -> list[str]:
    titles = []

    for value in values:
        try:
            genre_id = int(value)
            titles.append(genre_map.get(genre_id, str(genre_id)))
        except Exception:
            continue

    return titles


def load_tracks() -> pd.DataFrame:
    genre_map = load_genre_map()

    tracks = pd.read_csv(
        RAW_DIR / "tracks.csv",
        header=[0, 1],
        index_col=0,
    )

    out = pd.DataFrame(index=tracks.index)
    out.index.name = "track_id"

    out["title"] = tracks[("track", "title")]
    out["artist_name"] = tracks[("artist", "name")]
    out["album_title"] = tracks[("album", "title")]
    out["genre_top"] = tracks[("track", "genre_top")]
    out["language_code"] = tracks[("track", "language_code")]

    out["duration"] = pd.to_numeric(
        tracks[("track", "duration")],
        errors="coerce",
    )

    # Chỉ dùng để tính popularity, không nhất thiết lưu ra bảng cuối.
    out["listens"] = pd.to_numeric(
        tracks[("track", "listens")],
        errors="coerce",
    ).fillna(0)

    out["favorites"] = pd.to_numeric(
        tracks[("track", "favorites")],
        errors="coerce",
    ).fillna(0)

    out["interest"] = pd.to_numeric(
        tracks[("track", "interest")],
        errors="coerce",
    ).fillna(0)

    genres = tracks[("track", "genres")].apply(parse_list)
    genres_all = tracks[("track", "genres_all")].apply(parse_list)

    out["genres_titles"] = genres.apply(
        lambda xs: genre_ids_to_titles(xs, genre_map)
    )

    out["genres_all_titles"] = genres_all.apply(
        lambda xs: genre_ids_to_titles(xs, genre_map)
    )

    return out.reset_index()


def load_echonest_features() -> pd.DataFrame:
    echonest = pd.read_csv(
        RAW_DIR / "echonest.csv",
        header=[0, 1, 2],
        index_col=0,
    )

    out = pd.DataFrame(index=echonest.index)
    out.index.name = "track_id"

    audio_map = {
        "acousticness": ("echonest", "audio_features", "acousticness"),
        "danceability": ("echonest", "audio_features", "danceability"),
        "energy": ("echonest", "audio_features", "energy"),
        "instrumentalness": ("echonest", "audio_features", "instrumentalness"),
        "liveness": ("echonest", "audio_features", "liveness"),
        "speechiness": ("echonest", "audio_features", "speechiness"),
        "tempo": ("echonest", "audio_features", "tempo"),
        "valence": ("echonest", "audio_features", "valence"),
    }

    for output_col, source_col in audio_map.items():
        if source_col in echonest.columns:
            out[output_col] = pd.to_numeric(
                echonest[source_col],
                errors="coerce",
            )

    return out.reset_index()


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    audio_cols = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "valence",
    ]

    for col in audio_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(out[col].median()).fillna(0.5)

    bounded_cols = [
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "valence",
    ]

    for col in bounded_cols:
        if col in out.columns:
            out[col] = out[col].clip(0, 1)

    # tempo là BPM nên cần chuẩn hóa riêng để cosine dùng cùng thang 0-1.
    if "tempo" in out.columns:
        scaler = MinMaxScaler()
        out["tempo_norm"] = scaler.fit_transform(out[["tempo"]])
    else:
        out["tempo_norm"] = 0.5

    if "duration" in out.columns:
        out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
        out["duration"] = out["duration"].fillna(out["duration"].median())

        scaler = MinMaxScaler()
        out["duration_norm"] = scaler.fit_transform(out[["duration"]])
    else:
        out["duration_norm"] = 0.5

    return out


def limit_songs(songs: pd.DataFrame, song_limit: int | None) -> pd.DataFrame:
    """
    Giới hạn số bài hát nhưng không dùng subset_rank.
    Lấy tương đối đều theo genre_top để dữ liệu demo đa dạng hơn.
    """
    if not song_limit or len(songs) <= song_limit:
        return songs

    n_genres = songs["genre_top"].nunique()
    per_genre = max(1, song_limit // n_genres)

    sampled_parts = []
    for _, group in songs.groupby("genre_top", sort=False):
        sampled_parts.append(
            group.sample(
                min(len(group), per_genre),
                random_state=42,
            )
        )

    sampled = (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1, random_state=42)
        .head(song_limit)
        .reset_index(drop=True)
    )

    return sampled


def preprocess(song_limit: int = DEFAULT_SONG_LIMIT) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks()
    echonest = load_echonest_features()

    songs = tracks.merge(echonest, on="track_id", how="left")

    songs = songs[songs["genre_top"].notna()].copy()
    songs = songs[songs["title"].notna()].copy()

    songs["language_code"] = songs["language_code"].fillna("unknown")

    songs["genres_titles_text"] = songs["genres_titles"].apply(
        lambda xs: "|".join(str(x) for x in xs)
    )

    songs["genres_all_titles_text"] = songs["genres_all_titles"].apply(
        lambda xs: "|".join(str(x) for x in xs)
    )

    songs = limit_songs(songs, song_limit)
    songs = add_model_features(songs)

    keep_cols = [
        "title",
        "artist_name",
        "album_title",
        "genre_top",
        "genres_titles_text",
        "genres_all_titles_text",
        "language_code",
        "duration",
        "duration_norm",
        "acousticness",
        "danceability",
        "energy",
        "instrumentalness",
        "liveness",
        "speechiness",
        "tempo",
        "tempo_norm",
        "valence",
    ]

    songs = songs[[col for col in keep_cols if col in songs.columns]]
    songs = songs.reset_index(drop=True)

    from database import import_songs_df

    import_songs_df(songs)

    return songs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--song-limit", type=int, default=DEFAULT_SONG_LIMIT)
    args = parser.parse_args()

    songs = preprocess(args.song_limit)

    from config import DB_PATH

    print(f"Saved {len(songs)} songs to SQLite: {DB_PATH}")
    print("Top genres:")
    print(songs["genre_top"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()