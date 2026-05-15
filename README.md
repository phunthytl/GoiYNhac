# Hệ gợi ý nhạc dựa trên thông tin cá nhân

Project xây dựng hệ gợi ý nhạc cá nhân hóa từ FMA metadata, hồ sơ người dùng và dữ liệu nghe nhạc mô phỏng. Phiên bản hiện tại dùng implicit feedback: hệ thống học từ trạng thái `đã nghe/chưa nghe`, không còn dự đoán rating 1–5.

## 1. Dữ liệu

Dataset gốc: FMA - Free Music Archive.

Các nguồn chính:

```text
fma_metadata/
├── tracks.csv
├── genres.csv
├── echonest.csv
└── ...
```

FMA cung cấp metadata và audio features của bài hát, nhưng không có đầy đủ hồ sơ user và lịch sử nghe cá nhân. Vì vậy project sinh thêm dữ liệu user và interaction mô phỏng trong SQLite.

## 2. Hướng mô hình hiện tại

Bài toán được chuyển thành phân loại nhị phân:

```text
user + song -> listened
```

Trong đó:

```text
1 = user đã nghe / bài phù hợp
0 = user chưa nghe / negative sample
```

Hai mô hình đang dùng:

```text
LightGBMClassifier
RandomForestClassifier
```

Khi gợi ý, hệ thống lấy các bài user chưa nghe, dự đoán xác suất `listened = 1`, sau đó sắp xếp giảm dần theo `match_score` để lấy Top-N bài hát.

## 3. Schema chính

```text
accounts          Tài khoản đăng nhập
user_profiles     Hồ sơ cá nhân và sở thích âm thanh
songs             Bài hát đã tiền xử lý
interactions      Lịch sử bài user đã nghe
training_pairs    Dữ liệu train gồm positive/negative samples
```

`interactions` chỉ lưu bài đã nghe:

```text
user_id
song_id
listened = 1
```

`training_pairs` có cả hai lớp:

```text
listened = 1   positive sample
listened = 0   negative sample
```

## 4. Các file chính

```text
config.py                    Cấu hình path, random state, TARGET=listened
preprocess_songs.py          Xử lý FMA metadata/Echonest features và lưu songs vào SQLite
generate_synthetic_data.py   Sinh user, lịch sử đã nghe và negative samples
model/train_lightgbm.py      Train LightGBMClassifier
model/train_random_forest.py Train RandomForestClassifier
recommend.py                 Gợi ý Top-N theo match_score
app.py                       Web demo Flask
templates/                   Giao diện login, home, profile, recommendations, admin
```

## 5. Tiền xử lý bài hát

Tiền xử lý dùng trực tiếp audio features từ `echonest.csv`:

```text
energy
valence
danceability
tempo
acousticness
instrumentalness
liveness
speechiness
```

Các cột lưu trong bảng `songs` được lọc gọn, chỉ giữ metadata và feature cần cho app/model. Các feature thô không dùng như `mfcc_*`, `spectral_*`, `subset`, `hotttness`, `listens`, `favorites`, `interest` đã được bỏ khỏi DB sau khi tạo feature tổng hợp như `popularity`.

## 6. Sinh dữ liệu mô phỏng

Lệnh hiện tại:

```bash
python generate_synthetic_data.py --users 3000 --total-interactions 100000 --negative-ratio 1.0
```

Kết quả gần nhất:

```text
users: 3000
songs: 8801
interactions: 100000
training_pairs: 200000
```

Phân phối label trong `training_pairs`:

```text
listened=0: 100000
listened=1: 100000
```

Ý nghĩa:

- `interactions`: chỉ chứa bài user đã nghe.
- `training_pairs`: gồm 100k positive + 100k negative để train classifier.
- Không còn rating 1–5.
- Không còn cột `liked`.

## 7. Train/test split

Không dùng random row split toàn cục nữa.

Hiện dùng per-user holdout:

```text
Mỗi user:
- positive listened=1 được chia train/test
- negative listened=0 được chia train/test
```

Cách chia này phù hợp hơn với bài toán gợi ý vì test mô phỏng tình huống: với một user đã có một phần lịch sử nghe, model cần xếp hạng các bài phù hợp còn lại.

## 8. Kết quả model hiện tại

Dữ liệu train gần nhất:

```text
rows: 200000
train_size: 159968
test_size: 40032
```

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBMClassifier | 0.7283 | 0.7459 | 0.6925 | 0.7182 | 0.8028 | 0.6597 | 0.7985 | 0.8440 |
| RandomForestClassifier | 0.6483 | 0.6851 | 0.5489 | 0.6095 | 0.7277 | 0.6079 | 0.7506 | 0.7644 |

Nhận xét:

- LightGBMClassifier đang tốt hơn RandomForestClassifier trên cả F1, ROC-AUC và NDCG@10.
- `match_score` là xác suất model dự đoán user có khả năng nghe bài đó.
- Các metrics RMSE/MAE đã bỏ vì không còn bài toán hồi quy rating.

## 9. Chạy pipeline

```bash
python preprocess_songs.py --song-limit 20000
python generate_synthetic_data.py --users 3000 --total-interactions 100000 --negative-ratio 1.0
python model/train_lightgbm.py
python model/train_random_forest.py
```

## 10. Gợi ý nhạc CLI

```bash
python recommend.py --user-id 1 --top-k 10
```

Lọc theo genre:

```bash
python recommend.py --user-id 1 --top-k 10 --genre Pop
```

Output có `match_score` trong khoảng 0–1.

## 11. Chạy web demo

```bash
python app.py
```

Mở:

```text
http://127.0.0.1:5000
```

Demo login:

```text
user0001 / User@123456
```

Các trang chính:

```text
/login              Đăng nhập
/register           Đăng ký
/setup              Nhập thông tin cá nhân và sở thích âm thanh
/home               Duyệt danh sách bài hát
/recommendations    Gợi ý Top-N theo match_score
/profile            Hồ sơ user, bài đã nghe, cập nhật thông tin bằng modal
/admin              Dashboard dữ liệu/model
```

## 12. Hướng phát triển tiếp

- Ghi nhận hành vi nghe/click thật từ người dùng thay vì chỉ dùng dữ liệu mô phỏng.
- Bổ sung thời gian nghe, số lần nghe lại, skip, favorite để làm implicit feedback phong phú hơn.
- Thêm giải thích gợi ý dựa trên genre/audio features.
- Tối ưu negative sampling để phản ánh hành vi thực tế hơn.
- Thử thêm mô hình ranking chuyên biệt như LambdaMART hoặc learning-to-rank.
