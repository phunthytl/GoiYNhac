# Hệ gợi ý nhạc dựa trên thông tin cá nhân

Project sử dụng FMA metadata làm dữ liệu bài hát và sinh thêm dữ liệu người dùng/tương tác giả lập có kiểm soát để xây dựng hệ gợi ý nhạc cá nhân hóa.

## 1. Dữ liệu

Dataset gốc: FMA - Free Music Archive

Thư mục dữ liệu đã extract:

```text
fma_metadata/
├── tracks.csv
├── genres.csv
├── features.csv
├── echonest.csv
└── ...
```

FMA có metadata và audio features của bài hát, nhưng không có đầy đủ thông tin cá nhân/rating người dùng. Vì vậy project tạo thêm:

```text
data/musics.db
```

## 2. Model hiện tại

Đã triển khai hai mô hình để so sánh:

```text
RandomForestRegressor
LightGBMRegressor
```

Mục tiêu model:

```text
Dự đoán rating / preference score giữa user và bài hát.
```

Sau đó xếp hạng các bài hát theo `predicted_rating` để lấy Top-N recommendation.

## 3. Các file chính

```text
config.py                    Cấu hình path và tham số mặc định
preprocess_songs.py          Xử lý FMA metadata/features và lưu bảng songs vào SQLite
generate_synthetic_data.py   Sinh interaction/rating giả lập và lưu training_pairs vào SQLite
model/train_random_forest.py Train RandomForestRegressor và đánh giá
model/train_lightgbm.py      Train LightGBMRegressor và đánh giá
recommend.py                 Gợi ý Top-N bài hát cho một user
app.py                       Web demo Flask
MODEL_NOTES.md               Ghi chú lựa chọn model
BAO_CAO_HE_GOI_Y_NHAC.md     Báo cáo đầy đủ từ lý thuyết đến triển khai
```

Output sau khi chạy:

```text
data/musics.db
model/random_forest_recommender.joblib
model/lightgbm_recommender.joblib
reports/random_forest_metrics.json
reports/lightgbm_metrics.json
```

## 4. Cài thư viện

```bash
pip install -r requirements.txt
```

## 5. Chạy pipeline

```bash
python preprocess_songs.py --song-limit 20000
python generate_synthetic_data.py --users 1200 --interactions-per-user 35
python model/train_random_forest.py
python model/train_lightgbm.py
```

## 6. Gợi ý nhạc cho user

```bash
python recommend.py --user-id 1 --top-k 10
```

Lọc theo genre:

```bash
python recommend.py --user-id 1 --top-k 10 --genre Pop
```

## 7. Kết quả model hiện tại

Kết quả gần nhất:

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| RandomForestRegressor | 0.46724221100318813 | 0.3888330715761068 | 0.9406250000000002 | 0.8144322581953075 | 0.9872160730141202 |
| LightGBMRegressor | 0.47460454020624315 | 0.38197338505983425 | 0.9416666666666668 | 0.8153886281716684 | 0.9875261240560492 |

Cả hai mô hình dùng 63794 rows, train size 51035, test size 12759 và đánh giá trên 384 users.

## 8. Chạy web demo

```bash
python app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:5000
```

Các trang chính:

```text
/login              Đăng nhập tài khoản thật
/home               Duyệt danh sách bài hát
/recommendations    Gợi ý Top-N bằng LightGBMRegressor
/profile            Hồ sơ user, lịch sử interaction và cập nhật thông tin bằng modal
/admin              Dashboard dữ liệu/model
```

## 9. Hướng tiếp theo

- Thêm báo cáo EDA dữ liệu nhạc/user/interaction.
- Cải thiện cách sinh user profile/rating để dữ liệu cân bằng hơn.
- Bổ sung feature importance để giải thích mô hình.
- Ghi nhận interaction thật từ hành vi nghe/click/rating của người dùng để thay thế dần dữ liệu mô phỏng.
