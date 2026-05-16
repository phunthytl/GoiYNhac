from __future__ import annotations

"""
File app.py là điểm chạy chính của web demo hệ gợi ý nhạc.

Vai trò của file này:
1. Khởi tạo Flask app và session đăng nhập.
2. Kết nối với database SQLite thông qua các hàm trong database.py.
3. Load model machine learning đã train sẵn từ thư mục model/.
4. Nhận request từ người dùng qua các route web.
5. Tạo đặc trưng cho từng cặp user-song rồi gọi model để tính match_score.
6. Render dữ liệu ra các template HTML trong thư mục templates/.

Các phần xử lý nặng như tiền xử lý dữ liệu, sinh dữ liệu mô phỏng và train model
không nằm trong file này. app.py chỉ dùng kết quả đã có: database + model .joblib.
"""

from functools import wraps
import sqlite3
import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, session, url_for

from config import DATA_DIR, MODEL_DIR, RAW_DIR
from database import *
from recommend import build_candidate_features, FEATURES

# Tạo đối tượng Flask. __name__ giúp Flask xác định vị trí project hiện tại
# để tự tìm được thư mục templates/ và static/.
app = Flask(__name__)

# Khóa bí mật dùng để ký dữ liệu session và flash message.
# Trong project demo có thể hard-code như hiện tại; nếu deploy thật nên đưa
# giá trị này vào biến môi trường để tránh lộ secret trong source code.
app.secret_key = "music-recommender-demo-secret"

# Đảm bảo database đã có đầy đủ bảng cần thiết trước khi web app phục vụ request.
# Hàm init_db() sẽ tạo bảng nếu chưa có và xử lý một số migration schema cũ.
init_db()

# ---------------------------------------------------------------------------
# Các hàm hỗ trợ nạp dữ liệu từ database/model/report
# ---------------------------------------------------------------------------

def load_users() -> pd.DataFrame:
    """
    Đọc toàn bộ bảng user_profiles từ SQLite.

    Kết quả trả về là DataFrame, mỗi dòng là một user profile gồm thông tin
    cá nhân, genre yêu thích, ngôn ngữ yêu thích và các chỉ số gu âm thanh.
    app.py dùng dữ liệu này để tìm user hiện tại và tạo feature gợi ý.
    """
    return load_users_df()

def load_songs() -> pd.DataFrame:
    """
    Đọc toàn bộ bảng songs từ SQLite.

    Bảng songs được tạo bởi preprocess_songs.py, chứa metadata bài hát và
    audio features đã chuẩn hóa như energy, valence, danceability, tempo_norm.
    """
    return load_songs_df()

def load_interactions() -> pd.DataFrame:
    """
    Đọc bảng interactions từ SQLite.

    Bảng này chỉ lưu các bài hát mà user đã nghe. Dữ liệu này được dùng để:
    - hiển thị trạng thái "Đã nghe" trên giao diện;
    - loại bài đã nghe khỏi danh sách gợi ý;
    - hiển thị lịch sử nghe trong trang profile.
    """
    return load_interactions_df()

def load_model(model_name: str = "lightgbm"):
    """
    Load pipeline machine learning đã huấn luyện từ file .joblib.

    model_name:
    - "lightgbm": dùng LightGBMClassifier, là model ưu tiên trong app.
    - "random_forest": dùng RandomForestClassifier để so sánh kết quả.

    File .joblib lưu cả bước tiền xử lý feature lẫn model, nên khi dự đoán
    app chỉ cần truyền DataFrame có đúng các cột trong FEATURES.
    """
    if model_name == "random_forest":
        return joblib.load(MODEL_DIR / "random_forest_recommender.joblib")
    return joblib.load(MODEL_DIR / "lightgbm_recommender.joblib")

def load_metrics_file(filename: str) -> dict | None:
    """
    Đọc file metrics JSON của model trong thư mục reports/.

    Các file này được sinh sau khi train model, ví dụ:
    - lightgbm_metrics.json
    - random_forest_metrics.json

    Nếu file chưa tồn tại, trả về None để trang admin biết chưa có metrics.
    """
    path = DATA_DIR.parent / "reports" / filename
    if not path.exists():
        return None
    return pd.read_json(path, typ="series").to_dict()

# ---------------------------------------------------------------------------
# Xử lý hiển thị ngôn ngữ
# ---------------------------------------------------------------------------

LANGUAGE_LABELS = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "tr": "Turkish",
    "pl": "Polish", "nl": "Dutch", "sv": "Swedish", "bg": "Bulgarian",
    "el": "Greek", "he": "Hebrew", "sw": "Swahili", "ko": "Korean",
    "vi": "Vietnamese", "th": "Thai", "ms": "Malay", "id": "Indonesian",
    "unknown": "Không xác định",
}

# Chuyển mã ngôn ngữ thành tên dễ đọc trên giao diện.
def language_name(code: str) -> str:
    code = str(code) if pd.notna(code) else "unknown"
    return LANGUAGE_LABELS.get(code, code.upper())

# Đăng ký custom filter cho Jinja2
app.jinja_env.filters["language_name"] = language_name

# Quy đổi tuổi số thành nhóm tuổi dạng categorical feature.
def make_age_group(age: int) -> str:
    if age < 20: return "teen"
    if age < 30: return "young_adult"
    if age < 45: return "adult"
    return "mature"

# Decorator bảo vệ các route cần đăng nhập
def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "account_id" not in session or "user_id" not in session:
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapper

# Lấy profile nhạc của user đang đăng nhập
def current_user() -> pd.Series | None:
    if "user_id" not in session:
        return None
    return get_profile(int(session["user_id"]))

# Phân trang
def paginate_df(df: pd.DataFrame, page: int, per_page: int):
    total = len(df)
    total_pages = max((total + per_page - 1) // per_page, 1)
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return df.iloc[start:end], total, total_pages, page

# Tách chuỗi thành list
def split_genre_text(text: str) -> list[str]:
    return [g.strip() for g in str(text).split("|") if g and g.strip()]

# Tạo mask boolean cho các bài hát khớp với genre đang lọc.
def song_matches_genre_filter(df: pd.DataFrame, genre: str) -> pd.Series:
    wanted = genre.lower()
    top_match = df["genre_top"].fillna("").str.lower() == wanted
    direct_match = pd.Series(False, index=df.index)
    all_match = pd.Series(False, index=df.index)
    if "genres_titles_text" in df.columns:
        direct_match = df["genres_titles_text"].fillna("").str.lower().str.split("|").apply(lambda xs: wanted in xs)
    if "genres_all_titles_text" in df.columns:
        all_match = df["genres_all_titles_text"].fillna("").str.lower().str.split("|").apply(lambda xs: wanted in xs)
    return top_match | direct_match | all_match

# Lấy danh sách genre chính đang xuất hiện trong bảng songs để hiển thị filter trên giao diện.
def top_level_genres(songs: pd.DataFrame) -> list[str]:
    return sorted(songs["genre_top"].dropna().astype(str).unique().tolist())

# Lấy toàn bộ genre có thể dùng để lọc hoặc chọn trong giao diện, bao gồm cả genre chính và phụ.
def available_genres(songs: pd.DataFrame) -> list[str]:
    genres = set(top_level_genres(songs))
    for col in ["genres_titles_text", "genres_all_titles_text"]:
        if col in songs.columns:
            for text in songs[col].fillna(""):
                genres.update(split_genre_text(text))
    return sorted(genres)

def genre_children_by_parent(songs: pd.DataFrame) -> dict[str, list[str]]:
    present_genres = set(available_genres(songs))
    main_genres = set(top_level_genres(songs))
    genres_df = pd.read_csv(RAW_DIR / "genres.csv")
    id_to_title = dict(zip(genres_df["genre_id"].astype(int), genres_df["title"].astype(str)))
    children: dict[str, list[str]] = {g: [] for g in main_genres}
    for _, row in genres_df.iterrows():
        parent_id = row.get("parent", row.get("genre_parent_id"))
        if pd.isna(parent_id): continue
        try:
            parent_title = id_to_title.get(int(parent_id))
        except Exception: continue
        child_title = str(row["title"])
        if parent_title in main_genres and child_title in present_genres and child_title != parent_title:
            children.setdefault(parent_title, []).append(child_title)
    return {parent: sorted(set(items)) for parent, items in children.items() if items}

def get_recommendations_for_user(
    user_id: int,
    top_k: int = 12,
    genre: str | None = None,
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    # Load dữ liệu mới nhất từ database ở mỗi request
    users = load_users()
    songs = load_songs()
    interactions = load_interactions()
    model = load_model(model_name)

    # Lấy hồ sơ người dùng
    user_rows = users[users["user_id"] == int(user_id)]
    if user_rows.empty: return pd.DataFrame()
    user = user_rows.iloc[0]

    # Lọc bỏ các bài user đã nghe
    seen_song_ids = set(interactions.loc[interactions["user_id"] == user_id, "song_id"].astype(int))
    candidates = songs[~songs["id"].astype(int).isin(seen_song_ids)].copy()

    # Nếu có bộ lọc genre, chỉ giữ các bài thuộc genre đó
    if genre:
        candidates = candidates[song_matches_genre_filter(candidates, genre)].copy()
    if candidates.empty: return pd.DataFrame()

    # Feature engineering: mỗi dòng ứng viên được ghép thêm thông tin user và các điểm tương đồng
    feature_df = build_candidate_features(user, candidates)
    
    # predict_proba trả về cột xác suất [class 1]
    feature_df["match_score"] = model.predict_proba(feature_df[FEATURES])[:, 1]
    
    # Trả các cột cần hiển thị trên giao diện
    cols = [
        "id", "title", "artist_name", "album_title", "genre_top", "language_code",
        "duration", "energy", "valence", "danceability", "tempo", "tempo_norm",
        "acousticness", "instrumentalness", "liveness", "speechiness", "match_score",
    ]
    return feature_df.sort_values("match_score", ascending=False).head(top_k)[cols]

# Các route điều hướng web

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        account = verify_account(username, password)
        if account is None:
            flash("Tên đăng nhập hoặc mật khẩu không đúng.")
            return render_template("login.html", username=username)
        
        user_id = create_profile_for_account(int(account["id"]))

        session.clear()
        session["account_id"] = int(account["id"])
        session["username"] = account["username"]
        session["user_id"] = user_id
        flash(f"Đăng nhập thành công. Xin chào {account['username']}.")
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Vui lòng nhập tên đăng nhập và mật khẩu.")
            return render_template("register.html", username=username)
        if len(password) < 6:
            flash("Mật khẩu cần ít nhất 6 ký tự.")
            return render_template("register.html", username=username)
        if password != confirm_password:
            flash("Mật khẩu xác nhận không khớp.")
            return render_template("register.html", username=username)

        try:
            account_id = create_account(username, password)
        except sqlite3.IntegrityError:
            flash("Tên đăng nhập đã tồn tại.")
            return render_template("register.html", username=username)

        user_id = create_profile_for_account(account_id)
        session.clear()
        session["account_id"] = account_id
        session["username"] = username
        session["user_id"] = user_id
        flash("Đăng ký thành công. Hãy nhập thông tin cơ bản để cá nhân hóa gợi ý.")
        return redirect(url_for("setup"))
    return render_template("register.html")

@app.route("/setup", methods=["GET", "POST"])
@login_required
def setup():
    songs = load_songs()
    user_id = int(session["user_id"])
    user = current_user()
    if user is None:
        flash("Không tìm thấy user hiện tại.")
        return redirect(url_for("login"))

    main_genres = top_level_genres(songs)
    genre_children = genre_children_by_parent(songs)

    language_codes = sorted(songs["language_code"].fillna("unknown").unique().tolist())
    languages = [(code, language_name(code)) for code in language_codes]

    if request.method == "POST":
        age = int(request.form.get("age", 22))
        favorite_genres = request.form.getlist("favorite_genres")
        favorite_detailed_genres = request.form.getlist("favorite_detailed_genres")

        if not favorite_genres: favorite_genres = ["Pop"]
        
        update_profile(
            user_id,
            {
                "age": age,
                "age_group": make_age_group(age),
                "gender": request.form.get("gender", "Other"),
                "favorite_genres": "|".join(favorite_genres),
                "favorite_detailed_genres": "|".join(favorite_detailed_genres),
                "language_preference": request.form.get("language_preference", "en"),
                "preferred_energy": float(request.form.get("preferred_energy", 0.5)),
                "preferred_valence": float(request.form.get("preferred_valence", 0.5)),
                "preferred_danceability": float(request.form.get("preferred_danceability", 0.5)),
                "preferred_tempo": float(request.form.get("preferred_tempo", 0.5)),
                "preferred_acousticness": float(request.form.get("preferred_acousticness", 0.5)),
                "preferred_instrumentalness": float(request.form.get("preferred_instrumentalness", 0.5)),
                "preferred_liveness": float(request.form.get("preferred_liveness", 0.5)),
                "preferred_speechiness": float(request.form.get("preferred_speechiness", 0.5)),
            },
        )
        flash("Đã lưu hồ sơ cá nhân. Gợi ý sẽ dùng thông tin mới này.")
        return redirect(url_for("profile"))

    selected_genres = set(str(user.get("favorite_genres", "")).split("|"))
    selected_detail_genres = set(str(user.get("favorite_detailed_genres", "")).split("|"))
    return render_template(
        "setup.html", user=user, main_genres=main_genres,
        genre_children=genre_children, languages=languages,
        selected_genres=selected_genres, selected_detail_genres=selected_detail_genres,
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/home")
@login_required
def home():
    songs = load_songs()

    q = request.args.get("q", "").strip()
    genre = request.args.get("genre", "").strip()
    page = int(request.args.get("page", 1))
    per_page = 24

    filtered = songs.copy()
    if q:
        query = q.lower()
        filtered = filtered[
            filtered["title"].fillna("").str.lower().str.contains(query)
            | filtered["artist_name"].fillna("").str.lower().str.contains(query)
            | filtered["album_title"].fillna("").str.lower().str.contains(query)
        ]

    if genre:
        filtered = filtered[song_matches_genre_filter(filtered, genre)]

    sort_cols = [col for col in ["title"] if col in filtered.columns]
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=True)

    page_items, total, total_pages, page = paginate_df(filtered, page, per_page)
    genres = available_genres(songs)
    user = current_user()
    interactions = load_interactions()

    listened_song_ids = set(interactions.loc[interactions["user_id"] == int(session["user_id"]), "song_id"].astype(int))

    return render_template(
        "home.html", user=user, songs=page_items.to_dict("records"),
        listened_song_ids=listened_song_ids, genres=genres,
        q=q, genre=genre, page=page, total_pages=total_pages, total=total,
    )

@app.route("/recommendations")
@login_required
def recommendations():
    top_k = int(request.args.get("top_k", 10))

    if top_k not in {10, 15, 20}: top_k = 10
    user = current_user()
    interactions = load_interactions()
    listened_song_ids = set(interactions.loc[interactions["user_id"] == int(session["user_id"]), "song_id"].astype(int))
    
    lightgbm_recs = get_recommendations_for_user(
        int(session["user_id"]), top_k=top_k, model_name="lightgbm",
    )

    random_forest_recs = get_recommendations_for_user(
        int(session["user_id"]), top_k=top_k, model_name="random_forest",
    )
    return render_template(
        "recommendations.html", user=user,
        lightgbm_recs=lightgbm_recs.to_dict("records"),
        random_forest_recs=random_forest_recs.to_dict("records"),
        listened_song_ids=listened_song_ids, top_k=top_k,
    )

@app.route("/song/<int:song_id>/listen", methods=["POST"])
@login_required
def listen_song(song_id: int):
    songs = load_songs()

    exists = not songs[songs["id"].astype(int) == song_id].empty
    if not exists:
        flash("Không tìm thấy bài hát.")
        return redirect(url_for("home"))

    user_id = int(session["user_id"])

    mark_song_listened(user_id, song_id)

    update_profile_from_listened_song(user_id, song_id)
    
    flash("Đã ghi nhận bài hát này là đã nghe và cập nhật hồ sơ gu nhạc.")

    next_url = request.form.get("next") or url_for("song_detail", song_id=song_id)
    return redirect(next_url)

@app.route("/song/<int:song_id>")
@login_required
def song_detail(song_id: int):
    songs = load_songs()
    interactions = load_interactions()

    rows = songs[songs["id"].astype(int) == song_id]
    if rows.empty:
        flash("Không tìm thấy bài hát.")
        return redirect(url_for("home"))
    song = rows.iloc[0]
    user_id = int(session["user_id"])

    user_interactions = interactions[(interactions["user_id"] == user_id) & (interactions["song_id"] == song_id)]
    has_listened = not user_interactions.empty
    return render_template(
        "song_detail.html", song=song, user=current_user(),
        interactions=user_interactions.to_dict("records"), has_listened=has_listened,
    )

@app.route("/profile")
@login_required
def profile():
    user = current_user()
    interactions = load_interactions()
    songs = load_songs()

    user_interactions = interactions[interactions["user_id"] == int(session["user_id"])]
    
    merged = user_interactions.merge(
        songs[["id", "title", "artist_name", "genre_top"]].rename(columns={"id": "song_id"}),
        on="song_id", how="left"
    )

    history = merged.sort_values("song_id", ascending=False).head(20)
    
    main_genres = top_level_genres(songs)
    genre_children = genre_children_by_parent(songs)
    language_codes = sorted(songs["language_code"].fillna("unknown").unique().tolist())
    languages = [(code, language_name(code)) for code in language_codes]
    selected_genres = set(str(user.get("favorite_genres", "")).split("|"))
    selected_detail_genres = set(str(user.get("favorite_detailed_genres", "")).split("|"))

    return render_template(
        "profile.html", user=user, history=history.to_dict("records"),
        main_genres=main_genres, genre_children=genre_children,
        languages=languages, selected_genres=selected_genres,
        selected_detail_genres=selected_detail_genres,
    )

@app.route("/admin")
def admin():
    users = load_users()
    songs = load_songs()
    interactions = load_interactions()
    
    lightgbm_metrics = load_metrics_file("lightgbm_metrics.json")
    random_forest_metrics = load_metrics_file("random_forest_metrics.json")

    model_metrics = [m for m in [lightgbm_metrics, random_forest_metrics] if m]
    
    best_model = None
    if model_metrics:
        best_model = sorted(model_metrics, key=lambda m: (-m.get("f1", 0), -m.get("ndcg_at_10", 0)))[0]

    stats = {
        "users": len(users),
        "songs": len(songs),
        "interactions": len(interactions),
        "genres": songs["genre_top"].nunique(),
        "listened": int(interactions["listened"].sum()) if "listened" in interactions.columns else len(interactions),
    }

    top_genres = songs["genre_top"].value_counts().sort_values(ascending=False).to_dict()

    metric_explanations = {
        "Accuracy": "Tỷ lệ dự đoán đúng đã nghe/chưa nghe trên tập kiểm tra.",
        "Precision": "Trong các bài model dự đoán là sẽ nghe, tỷ lệ dự đoán đúng.",
        "Recall": "Trong các bài thực sự đã nghe, tỷ lệ model tìm lại được.",
        "F1": "Trung bình điều hòa giữa Precision và Recall. Càng cao càng tốt.",
        "AUC": "Khả năng phân biệt bài đã nghe và chưa nghe. Càng gần 1 càng tốt.",
        "Precision@10": "Trong 10 bài được gợi ý đầu tiên, tỷ lệ bài thuộc nhóm đã nghe/phù hợp.",
        "Recall@10": "Trong các bài phù hợp của user, hệ thống tìm lại được bao nhiêu bài trong Top 10.",
        "NDCG@10": "Đánh giá chất lượng thứ hạng Top 10, ưu tiên bài phù hợp xuất hiện ở vị trí cao.",
    }

    return render_template(
        "admin.html", stats=stats, lightgbm_metrics=lightgbm_metrics,
        random_forest_metrics=random_forest_metrics, model_metrics=model_metrics,
        best_model=best_model, top_genres=top_genres, metric_explanations=metric_explanations,
    )

if __name__ == "__main__":
    app.run(debug=False)
