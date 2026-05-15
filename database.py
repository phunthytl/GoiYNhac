import sqlite3
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

from config import DB_PATH

PROFILE_COLUMNS = [
    "user_id",
    "account_id",
    "age",
    "age_group",
    "gender",
    "favorite_genres",
    "language_preference",
    "preferred_energy",
    "preferred_valence",
    "preferred_danceability",
    "preferred_tempo",
    "preferred_popularity",
    "preferred_acousticness",
    "preferred_instrumentalness",
    "preferred_liveness",
    "preferred_speechiness",
    "favorite_detailed_genres",
]

DEFAULT_PROFILE = {
    "age": 22,
    "age_group": "young_adult",
    "gender": "Other",
    "favorite_genres": "Pop|Electronic",
    "language_preference": "en",
    "preferred_energy": 0.5,
    "preferred_valence": 0.5,
    "preferred_danceability": 0.5,
    "preferred_tempo": 0.5,
    "preferred_popularity": 0.5,
    "preferred_acousticness": 0.5,
    "preferred_instrumentalness": 0.5,
    "preferred_liveness": 0.5,
    "preferred_speechiness": 0.5,
    "favorite_detailed_genres": "",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_user_profile_auth_columns(conn: sqlite3.Connection) -> None:
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(user_profiles)")}
    if "account_id" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN account_id INTEGER")
    if "favorite_detailed_genres" not in columns:
        conn.execute("ALTER TABLE user_profiles ADD COLUMN favorite_detailed_genres TEXT DEFAULT ''")
    for col in ["preferred_acousticness", "preferred_instrumentalness", "preferred_liveness", "preferred_speechiness"]:
        if col not in columns:
            conn.execute(f"ALTER TABLE user_profiles ADD COLUMN {col} REAL DEFAULT 0.5")


def table_has_integer_pk(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for row in rows:
        if row["name"] == column:
            return int(row["pk"] or 0) == 1 and str(row["type"]).upper() == "INTEGER"
    return False


def migrate_implicit_feedback_tables(conn: sqlite3.Connection) -> None:
    interaction_cols = {row["name"] for row in conn.execute("PRAGMA table_info(interactions)")}
    if "rating" in interaction_cols or (interaction_cols and "listened" not in interaction_cols):
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS interactions_fixed (
                user_id INTEGER NOT NULL,
                song_id INTEGER NOT NULL,
                listened INTEGER NOT NULL DEFAULT 1,
                created_at TEXT,
                PRIMARY KEY(user_id, song_id),
                FOREIGN KEY(user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                FOREIGN KEY(song_id) REFERENCES songs(id) ON DELETE CASCADE
            );
            INSERT OR IGNORE INTO interactions_fixed (user_id, song_id, listened, created_at)
            SELECT user_id, song_id, 1, NULL
            FROM interactions
            WHERE user_id IS NOT NULL AND song_id IS NOT NULL;
            DROP TABLE interactions;
            ALTER TABLE interactions_fixed RENAME TO interactions;
            """
        )
        conn.execute("PRAGMA foreign_keys = ON")

    pair_cols = {row["name"] for row in conn.execute("PRAGMA table_info(training_pairs)")}
    if "rating" in pair_cols or (pair_cols and "listened" not in pair_cols):
        conn.executescript(
            """
            DROP TABLE IF EXISTS training_pairs;
            CREATE TABLE training_pairs (
                user_id INTEGER,
                song_id INTEGER,
                listened INTEGER NOT NULL
            );
            """
        )


def repair_auth_tables(conn: sqlite3.Connection) -> None:
    """Sửa schema bị pandas to_sql(if_exists='replace') làm mất PRIMARY KEY/AUTOINCREMENT."""
    conn.execute("PRAGMA foreign_keys = OFF")

    if not table_has_integer_pk(conn, "accounts", "id"):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts_fixed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT OR IGNORE INTO accounts_fixed (id, username, password_hash, created_at)
            SELECT COALESCE(id, rowid), username, password_hash, COALESCE(created_at, '')
            FROM accounts
            WHERE username IS NOT NULL AND password_hash IS NOT NULL;
            DROP TABLE accounts;
            ALTER TABLE accounts_fixed RENAME TO accounts;
            """
        )

    if not table_has_integer_pk(conn, "user_profiles", "user_id"):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profiles_fixed (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER UNIQUE,
                age INTEGER NOT NULL,
                age_group TEXT NOT NULL,
                gender TEXT NOT NULL,
                favorite_genres TEXT NOT NULL,
                language_preference TEXT NOT NULL,
                preferred_energy REAL NOT NULL,
                preferred_valence REAL NOT NULL,
                preferred_danceability REAL NOT NULL,
                preferred_tempo REAL NOT NULL,
                preferred_popularity REAL NOT NULL,
                preferred_acousticness REAL DEFAULT 0.5,
                preferred_instrumentalness REAL DEFAULT 0.5,
                preferred_liveness REAL DEFAULT 0.5,
                preferred_speechiness REAL DEFAULT 0.5,
                favorite_detailed_genres TEXT DEFAULT '',
                FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE
            );
            INSERT OR IGNORE INTO user_profiles_fixed (
                user_id, account_id, age, age_group, gender, favorite_genres, language_preference,
                preferred_energy, preferred_valence, preferred_danceability, preferred_tempo,
                preferred_popularity, preferred_acousticness, preferred_instrumentalness,
                preferred_liveness, preferred_speechiness, favorite_detailed_genres
            )
            SELECT
                COALESCE(user_id, account_id, rowid),
                account_id,
                COALESCE(age, 22),
                COALESCE(age_group, 'young_adult'),
                COALESCE(gender, 'Other'),
                COALESCE(favorite_genres, 'Pop|Electronic'),
                COALESCE(language_preference, 'en'),
                COALESCE(preferred_energy, 0.5),
                COALESCE(preferred_valence, 0.5),
                COALESCE(preferred_danceability, 0.5),
                COALESCE(preferred_tempo, 0.5),
                COALESCE(preferred_popularity, 0.5),
                COALESCE(preferred_acousticness, 0.5),
                COALESCE(preferred_instrumentalness, 0.5),
                COALESCE(preferred_liveness, 0.5),
                COALESCE(preferred_speechiness, 0.5),
                COALESCE(favorite_detailed_genres, '')
            FROM user_profiles;
            DROP TABLE user_profiles;
            ALTER TABLE user_profiles_fixed RENAME TO user_profiles;
            """
        )

    conn.execute("PRAGMA foreign_keys = ON")


def init_db() -> None:
    with get_connection() as conn:
        # Kiểm tra và xóa bảng songs cũ nếu không có cột id
        try:
            cursor = conn.execute("PRAGMA table_info(songs)")
            columns = {row[1] for row in cursor.fetchall()}
            if "id" not in columns:
                conn.execute("DROP TABLE IF EXISTS songs")
                conn.execute("DROP TABLE IF EXISTS interactions")
                conn.execute("DROP TABLE IF EXISTS training_pairs")
        except Exception:
            pass
        
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER UNIQUE,
                age INTEGER NOT NULL,
                age_group TEXT NOT NULL,
                gender TEXT NOT NULL,
                favorite_genres TEXT NOT NULL,
                language_preference TEXT NOT NULL,
                preferred_energy REAL NOT NULL,
                preferred_valence REAL NOT NULL,
                preferred_danceability REAL NOT NULL,
                preferred_tempo REAL NOT NULL,
                preferred_popularity REAL NOT NULL,
                preferred_acousticness REAL DEFAULT 0.5,
                preferred_instrumentalness REAL DEFAULT 0.5,
                preferred_liveness REAL DEFAULT 0.5,
                preferred_speechiness REAL DEFAULT 0.5,
                favorite_detailed_genres TEXT DEFAULT '',
                FOREIGN KEY(account_id) REFERENCES accounts(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                artist_name TEXT,
                album_title TEXT,
                genre_top TEXT,
                genres_titles_text TEXT,
                genres_all_titles_text TEXT,
                language_code TEXT,
                duration REAL,
                duration_norm REAL,
                acousticness REAL,
                danceability REAL,
                energy REAL,
                instrumentalness REAL,
                liveness REAL,
                speechiness REAL,
                tempo REAL,
                tempo_norm REAL,
                valence REAL,
                popularity REAL
            );

            CREATE TABLE IF NOT EXISTS interactions (
                user_id INTEGER NOT NULL,
                song_id INTEGER NOT NULL,
                listened INTEGER NOT NULL DEFAULT 1,
                created_at TEXT,
                PRIMARY KEY(user_id, song_id),
                FOREIGN KEY(user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                FOREIGN KEY(song_id) REFERENCES songs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS training_pairs (
                user_id INTEGER,
                song_id INTEGER,
                listened INTEGER NOT NULL
            );
            """
        )
        ensure_user_profile_auth_columns(conn)
        repair_auth_tables(conn)
        migrate_implicit_feedback_tables(conn)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            )
            out[col] = out[col].where(out[col].notna(), None)
    return out


def import_dataframe(table: str, df: pd.DataFrame, if_exists: str = "replace") -> int:
    out = normalize_dataframe(df)
    if table == "user_profiles":
        for col in PROFILE_COLUMNS:
            if col not in out.columns:
                out[col] = None if col == "account_id" else ""
        out["favorite_detailed_genres"] = out["favorite_detailed_genres"].fillna("")
        out = out[PROFILE_COLUMNS]
    with get_connection() as conn:
        out.to_sql(table, conn, if_exists=if_exists, index=False)
    return len(out)


def import_songs_df(songs: pd.DataFrame) -> int:
    init_db()
    out = normalize_dataframe(songs).copy()
    # Thêm cột id tự động tăng từ 1
    out.insert(0, "id", range(1, len(out) + 1))
    with get_connection() as conn:
        # Tắt foreign_keys tạm thời để xóa dữ liệu
        conn.execute("PRAGMA foreign_keys = OFF")
        # Xóa dữ liệu cũ nhưng giữ lại schema
        conn.execute("DELETE FROM songs")
        # Reset AUTO_INCREMENT counter
        conn.execute("DELETE FROM sqlite_sequence WHERE name='songs'")
        # Insert dữ liệu mới với id
        out.to_sql("songs", conn, if_exists="append", index=False)
        # Bật lại foreign_keys
        conn.execute("PRAGMA foreign_keys = ON")
    return len(out)


def import_synthetic_users_with_accounts(users: pd.DataFrame, password: str = "123456") -> dict[str, int]:
    """Lưu user profile giả lập kèm account đăng nhập demo.

    Account được tạo theo mẫu user0001, user0002... và dùng chung mật khẩu demo.
    Mật khẩu vẫn được hash trước khi lưu DB.
    """
    init_db()
    accounts = []
    profiles = users.copy()
    profiles["account_id"] = profiles["user_id"].astype(int)
    if "favorite_detailed_genres" not in profiles.columns:
        profiles["favorite_detailed_genres"] = ""

    password_hash = generate_password_hash(password)
    for user_id in profiles["user_id"].astype(int):
        accounts.append(
            {
                "id": user_id,
                "username": f"user{user_id:04d}",
                "password_hash": password_hash,
                "created_at": utc_now(),
            }
        )

    import_dataframe("accounts", pd.DataFrame(accounts))
    import_dataframe("user_profiles", profiles)
    return {"accounts": len(accounts), "user_profiles": len(profiles)}


def import_training_data(interactions: pd.DataFrame, training_pairs: pd.DataFrame) -> dict[str, int]:
    init_db()
    counts = {
        "interactions": import_dataframe("interactions", interactions),
        "training_pairs": import_dataframe("training_pairs", training_pairs),
    }
    return counts


def table_exists(table: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
    return row is not None


def read_table(table: str) -> pd.DataFrame:
    init_db()
    with get_connection() as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def load_songs_df() -> pd.DataFrame:
    return read_table("songs")


def load_users_df() -> pd.DataFrame:
    return read_table("user_profiles")


def load_interactions_df() -> pd.DataFrame:
    return read_table("interactions")


def load_training_pairs_df() -> pd.DataFrame:
    return read_table("training_pairs")


def row_to_series(row: sqlite3.Row | None) -> pd.Series | None:
    if row is None:
        return None
    return pd.Series(dict(row))


def get_profile(user_id: int) -> pd.Series | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)).fetchone()
    return row_to_series(row)


def get_profile_by_account(account_id: int) -> pd.Series | None:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM user_profiles WHERE account_id = ?", (account_id,)).fetchone()
    return row_to_series(row)


def get_account(username: str) -> sqlite3.Row | None:
    init_db()
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM accounts WHERE lower(username) = lower(?)",
            (username.strip(),),
        ).fetchone()


def create_account(username: str, password: str) -> int:
    init_db()
    username = username.strip()
    password_hash = generate_password_hash(password)
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO accounts (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, utc_now()),
        )
        return int(cur.lastrowid)


def verify_account(username: str, password: str) -> sqlite3.Row | None:
    account = get_account(username)
    if account is None:
        return None
    if not check_password_hash(account["password_hash"], password):
        return None
    return account


def create_profile_for_account(account_id: int) -> int:
    init_db()
    profile = get_profile_by_account(account_id)
    if profile is not None and pd.notna(profile.get("user_id")):
        return int(profile["user_id"])

    values = DEFAULT_PROFILE.copy()
    with get_connection() as conn:
        conn.execute("DELETE FROM user_profiles WHERE account_id = ? AND user_id IS NULL", (account_id,))
        cur = conn.execute(
            """
            INSERT INTO user_profiles (
                account_id, age, age_group, gender, favorite_genres, language_preference,
                preferred_energy, preferred_valence, preferred_danceability, preferred_tempo,
                preferred_popularity, preferred_acousticness, preferred_instrumentalness,
                preferred_liveness, preferred_speechiness, favorite_detailed_genres
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                values["age"],
                values["age_group"],
                values["gender"],
                values["favorite_genres"],
                values["language_preference"],
                values["preferred_energy"],
                values["preferred_valence"],
                values["preferred_danceability"],
                values["preferred_tempo"],
                values["preferred_popularity"],
                values["preferred_acousticness"],
                values["preferred_instrumentalness"],
                values["preferred_liveness"],
                values["preferred_speechiness"],
                values["favorite_detailed_genres"],
            ),
        )
        return int(cur.lastrowid)


def update_profile(user_id: int, data: dict[str, Any]) -> None:
    allowed = [col for col in PROFILE_COLUMNS if col not in {"user_id", "account_id"}]
    fields = [key for key in allowed if key in data]
    if not fields:
        return
    assignments = ", ".join(f"{field} = ?" for field in fields)
    values = [data[field] for field in fields]
    values.append(user_id)
    init_db()
    with get_connection() as conn:
        conn.execute(f"UPDATE user_profiles SET {assignments} WHERE user_id = ?", values)