# CHƯƠNG 1. GIỚI THIỆU CHUNG

## 1.1. Bối cảnh đề tài

Trong thời đại số, các nền tảng nghe nhạc trực tuyến cung cấp kho nhạc rất lớn cho người dùng. Số lượng bài hát tăng nhanh giúp người dùng có nhiều lựa chọn hơn, nhưng đồng thời tạo ra vấn đề quá tải thông tin. Người dùng khó tự tìm được bài hát phù hợp với sở thích cá nhân, ngôn ngữ yêu thích, tâm trạng hoặc thói quen nghe nhạc của mình.

Hệ gợi ý nhạc được xây dựng nhằm hỗ trợ người dùng khám phá bài hát phù hợp hơn. Thay vì chỉ hiển thị danh sách bài hát phổ biến chung cho tất cả mọi người, hệ thống tận dụng hồ sơ người dùng và đặc trưng âm thanh của bài hát để tạo danh sách gợi ý cá nhân hóa.

Đề tài “Hệ gợi ý nhạc dựa trên thông tin cá nhân” tập trung xây dựng một hệ thống gợi ý sử dụng dữ liệu metadata và audio features từ bộ dữ liệu FMA, kết hợp với hồ sơ người dùng và dữ liệu nghe nhạc mô phỏng. Hệ thống được xây dựng theo hướng implicit feedback: học từ trạng thái bài hát đã nghe/chưa nghe và dự đoán xác suất người dùng có khả năng nghe một bài hát.

## 1.2. Mục tiêu đề tài

Mục tiêu của đề tài là xây dựng hệ gợi ý nhạc cá nhân hóa có khả năng sử dụng thông tin hồ sơ người dùng, sở thích âm thanh và đặc trưng bài hát để dự đoán mức độ phù hợp giữa người dùng và bài hát. Hệ thống xếp hạng các bài hát theo xác suất phù hợp và đề xuất danh sách Top-N cho từng người dùng.

Các mục tiêu cụ thể:

1. Tìm hiểu cơ sở lý thuyết về hệ gợi ý, đặc biệt là gợi ý nhạc cá nhân hóa dựa trên hồ sơ người dùng và implicit feedback.
2. Khai thác dữ liệu FMA metadata, tiền xử lý thông tin bài hát và đặc trưng âm thanh, sau đó lưu vào SQLite để phục vụ web app và mô hình.
3. Sinh dữ liệu người dùng và lịch sử nghe nhạc mô phỏng do FMA không cung cấp đầy đủ dữ liệu cá nhân/hành vi nghe.
4. Thiết kế bộ đặc trưng kết hợp giữa user profile, metadata bài hát và audio features.
5. Triển khai và so sánh hai mô hình học máy trên dữ liệu bảng: RandomForestClassifier và LightGBMClassifier.
6. Đánh giá mô hình bằng các chỉ số phân loại và xếp hạng như Accuracy, Precision, Recall, F1, ROC-AUC, Precision@10, Recall@10 và NDCG@10.
7. Xây dựng web demo Flask có đăng ký/đăng nhập, lưu hồ sơ cá nhân, cập nhật sở thích nghe nhạc và hiển thị gợi ý Top-N.

## 1.3. Bài toán nghiên cứu

Bài toán được đặt ra như sau:

- Với một người dùng có thông tin cá nhân và sở thích âm nhạc ban đầu, hệ thống cần dự đoán xác suất người dùng có khả năng nghe từng bài hát chưa nghe, sau đó xếp hạng và đề xuất các bài hát phù hợp nhất.

Đầu vào của bài toán:

- thông tin người dùng: tuổi, giới tính, nhóm tuổi;
- thể loại yêu thích;
- ngôn ngữ nhạc yêu thích;
- sở thích âm thanh: energy, valence, danceability, tempo, popularity, acousticness, instrumentalness, liveness, speechiness;
- metadata bài hát;
- đặc trưng âm thanh của bài hát.

Đầu ra của bài toán:

```text
match_score(user, song) = P(listened = 1 | user, song)
```

Trong đó:

```text
listened = 1: user đã nghe / bài phù hợp
listened = 0: user chưa nghe / negative sample
```

Sau khi có `match_score`, hệ thống sắp xếp các bài hát chưa nghe theo điểm giảm dần và lấy Top-N bài hát làm danh sách gợi ý.

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1. Khái niệm hệ gợi ý

Hệ gợi ý là hệ thống có khả năng đề xuất các đối tượng phù hợp với từng người dùng dựa trên dữ liệu đã thu thập được. Đối tượng được gợi ý có thể là phim, bài hát, sản phẩm, khóa học, bài viết hoặc video.

Trong bài toán gợi ý nhạc:

- User: người nghe nhạc.
- Item: bài hát.
- Interaction: hành vi giữa user và bài hát, ví dụ nghe, bỏ qua, yêu thích, thêm playlist.

Mục tiêu của hệ gợi ý không chỉ là tìm bài hát phổ biến nhất, mà là tìm bài hát phù hợp nhất với từng người dùng cụ thể.

## 2.2. Vai trò của hệ gợi ý trong ứng dụng nghe nhạc

Hệ gợi ý trong ứng dụng nghe nhạc có các vai trò:

- giúp người dùng khám phá bài hát mới;
- giảm thời gian tìm kiếm thủ công;
- cá nhân hóa trải nghiệm nghe nhạc;
- tăng mức độ tương tác;
- hỗ trợ xây dựng playlist tự động;
- giữ chân người dùng nhờ nội dung phù hợp hơn.

Nếu không có hệ gợi ý, người dùng thường chỉ thấy bảng xếp hạng chung. Danh sách này không phản ánh đầy đủ gu âm nhạc riêng của từng người.

## 2.3. Các thành phần dữ liệu

### 2.3.1. User

User được mô tả bằng thông tin cá nhân và sở thích âm thanh:

| Nhóm thông tin | Thuộc tính |
|---|---|
| Thông tin cá nhân | age, age_group, gender |
| Sở thích nội dung | favorite_genres, favorite_detailed_genres, language_preference |
| Sở thích âm thanh | preferred_energy, preferred_valence, preferred_danceability, preferred_tempo, preferred_popularity |
| Sở thích âm thanh mở rộng | preferred_acousticness, preferred_instrumentalness, preferred_liveness, preferred_speechiness |

### 2.3.2. Item

Item là bài hát. Mỗi bài hát gồm:

| Nhóm thông tin | Thuộc tính |
|---|---|
| Metadata | title, artist_name, album_title, genre_top, language_code, duration |
| Genre chi tiết | genres_titles_text, genres_all_titles_text |
| Audio features | energy, valence, danceability, tempo, tempo_norm |
| Audio features mở rộng | acousticness, instrumentalness, liveness, speechiness |
| Đặc trưng tổng hợp | duration_norm, popularity |

### 2.3.3. Interaction

Interaction là tương tác giữa user và bài hát. Trong đề tài này, interaction được biểu diễn theo implicit feedback:

```text
user_id, song_id, listened
```

Trong đó bảng `interactions` chỉ lưu bài user đã nghe:

```text
listened = 1
```

Để huấn luyện mô hình phân loại, bảng `training_pairs` có cả:

```text
listened = 1: positive sample
listened = 0: negative sample
```

## 2.4. Các hướng tiếp cận phổ biến

### 2.4.1. Popularity-Based Recommendation

Popularity-Based Recommendation đề xuất các bài hát phổ biến nhất cho tất cả người dùng.

Ưu điểm:

- dễ triển khai;
- không cần thông tin cá nhân;
- phù hợp khi chưa có dữ liệu user.

Nhược điểm:

- không cá nhân hóa;
- mọi user nhận kết quả giống nhau;
- không phản ánh gu âm nhạc riêng.

### 2.4.2. Content-Based Filtering

Content-Based Filtering dựa trên đặc trưng của bài hát như genre, language, energy, valence, danceability, tempo. Hệ thống ưu tiên các bài hát có đặc trưng gần với sở thích user.

Ưu điểm:

- phù hợp với user mới nếu user nhập sở thích ban đầu;
- có thể giải thích dựa trên nội dung bài hát;
- không phụ thuộc hoàn toàn vào hành vi cộng đồng.

Nhược điểm:

- dễ bị giới hạn trong vùng sở thích cũ;
- chất lượng phụ thuộc vào đặc trưng item;
- khó gợi ý bài khác gu nhưng user có thể thích.

### 2.4.3. Collaborative Filtering

Collaborative Filtering dựa trên hành vi của cộng đồng người dùng. Những người dùng có hành vi giống nhau có thể thích các bài hát giống nhau.

Ưu điểm:

- khai thác hành vi cộng đồng;
- có thể gợi ý bài ngoài sở thích khai báo ban đầu;
- không cần hiểu sâu nội dung bài hát.

Nhược điểm:

- khó xử lý user mới;
- khó xử lý bài hát mới;
- cần dữ liệu tương tác đủ lớn;
- dữ liệu thường thưa.

### 2.4.4. Profile-Based Recommendation

Profile-Based Recommendation sử dụng hồ sơ user như tuổi, giới tính, thể loại yêu thích, ngôn ngữ yêu thích và sở thích âm thanh. Cách tiếp cận này phù hợp với đề tài vì hệ thống cần gợi ý ngay cả khi user mới chưa có nhiều lịch sử nghe.

## 2.5. Bài toán cold-start

Cold-start xảy ra khi user mới hoặc bài hát mới chưa có đủ tương tác.

### 2.5.1. User cold-start

User mới chưa có lịch sử nghe. Hệ thống xử lý bằng cách yêu cầu user nhập:

- tuổi;
- giới tính;
- ngôn ngữ nhạc yêu thích;
- thể loại yêu thích;
- sở thích âm thanh.

Nhờ đó user mới vẫn có thể nhận gợi ý dựa trên hồ sơ.

### 2.5.2. Item cold-start

Bài hát mới chưa có tương tác. Hệ thống xử lý bằng metadata và audio features của bài hát.

## 2.6. Đặc trưng âm thanh

Project sử dụng trực tiếp các đặc trưng từ `echonest.csv` thay vì tự tính bằng công thức heuristic.

| Feature | Ý nghĩa |
|---|---|
| energy | mức độ mạnh, sôi động |
| valence | sắc thái cảm xúc tích cực |
| danceability | mức độ dễ nhún nhảy |
| tempo / tempo_norm | tốc độ bài hát, đã chuẩn hóa |
| popularity | mức độ phổ biến tổng hợp |
| acousticness | mức độ mộc/acoustic |
| instrumentalness | mức độ không lời |
| liveness | cảm giác biểu diễn trực tiếp |
| speechiness | mức độ lời/nói rõ |

## 2.7. Bài toán implicit feedback

Khác với explicit feedback, implicit feedback không yêu cầu user chấm điểm. Hệ thống học từ hành vi như đã nghe/chưa nghe, click, thêm playlist hoặc nghe lại.

Trong đề tài này, bài toán được mô hình hóa thành phân loại nhị phân:

```text
Input: user features + song features
Output: listened ∈ {0, 1}
```

Model dự đoán xác suất:

```text
P(listened = 1 | user, song)
```

Xác suất này được dùng làm `match_score` để xếp hạng bài hát.

## 2.8. Chỉ số đánh giá

### 2.8.1. Accuracy

Accuracy đo tỷ lệ dự đoán đúng trên tập kiểm tra.

```text
Accuracy = số dự đoán đúng / tổng số mẫu
```

### 2.8.2. Precision

Precision đo trong các bài model dự đoán là user sẽ nghe, có bao nhiêu bài đúng.

```text
Precision = TP / (TP + FP)
```

### 2.8.3. Recall

Recall đo trong các bài user thực sự nghe, model tìm lại được bao nhiêu bài.

```text
Recall = TP / (TP + FN)
```

### 2.8.4. F1-score

F1 là trung bình điều hòa giữa Precision và Recall.

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

### 2.8.5. ROC-AUC

ROC-AUC đo khả năng phân biệt giữa class đã nghe và chưa nghe. Giá trị càng gần 1 càng tốt.

### 2.8.6. Precision@k

Precision@k đo trong k bài gợi ý đầu tiên, có bao nhiêu bài thuộc nhóm phù hợp.

```text
Precision@k = số bài relevant trong Top k / k
```

### 2.8.7. Recall@k

Recall@k đo trong tất cả bài relevant của user, hệ thống tìm lại được bao nhiêu bài trong Top k.

```text
Recall@k = số bài relevant trong Top k / tổng số bài relevant
```

### 2.8.8. NDCG@k

NDCG@k đánh giá chất lượng thứ hạng. Bài phù hợp xuất hiện ở vị trí cao sẽ làm NDCG tăng.

## 2.9. Kết luận chương

Đề tài chọn hướng profile-based kết hợp content-based features và implicit feedback. Bài toán được đưa về phân loại nhị phân đã nghe/chưa nghe, sau đó dùng xác suất dự đoán để xếp hạng Top-N.

# CHƯƠNG 3. PHƯƠNG PHÁP ĐỀ XUẤT

## 3.1. Tổng quan phương pháp

Quy trình đề xuất gồm:

1. Đọc dữ liệu FMA metadata.
2. Tiền xử lý bài hát.
3. Lấy audio features từ Echonest.
4. Lưu dữ liệu bài hát vào SQLite.
5. Sinh hồ sơ user.
6. Sinh lịch sử bài đã nghe.
7. Sinh negative samples cho bài chưa nghe.
8. Tạo `training_pairs` dạng user-song-label.
9. Huấn luyện LightGBMClassifier và RandomForestClassifier.
10. Đánh giá bằng classification metrics và ranking metrics.
11. Triển khai web demo Flask.

## 3.2. Dữ liệu sử dụng

Bộ dữ liệu gốc: FMA - Free Music Archive.

Các file chính:

| File | Vai trò |
|---|---|
| tracks.csv | metadata bài hát, album, artist, genre |
| genres.csv | cây thể loại nhạc |
| echonest.csv | audio features như energy, valence, danceability |

FMA không cung cấp đầy đủ user profile và lịch sử nghe cá nhân, nên project sinh dữ liệu mô phỏng để phục vụ huấn luyện và demo.

## 3.3. Tiền xử lý dữ liệu bài hát

Các bước chính:

1. Đọc metadata từ `tracks.csv`.
2. Đọc cây thể loại từ `genres.csv`.
3. Đọc audio features từ `echonest.csv`.
4. Chuyển genre id sang tên genre.
5. Chuẩn hóa các cột số.
6. Tạo `tempo_norm`, `duration_norm`, `popularity`.
7. Lọc bỏ cột không dùng.
8. Lưu kết quả vào bảng `songs` trong SQLite.

Bảng `songs` sau xử lý giữ các cột chính:

```text
id, title, artist_name, album_title, genre_top,
genres_titles_text, genres_all_titles_text,
language_code, duration, duration_norm,
acousticness, danceability, energy, instrumentalness,
liveness, speechiness, tempo, tempo_norm, valence, popularity
```

Các cột low-level hoặc không còn dùng như `mfcc_*`, `spectral_*`, `subset`, `hotttness`, `listens`, `favorites`, `interest` không được lưu vào DB.

## 3.4. Sinh hồ sơ user và interaction

Do không có dữ liệu user thật, hệ thống sinh user mô phỏng.

Mỗi user có:

- tuổi;
- nhóm tuổi;
- giới tính;
- thể loại yêu thích;
- sub-genre yêu thích;
- ngôn ngữ nhạc yêu thích;
- sở thích âm thanh dạng slider 0–1.

Dữ liệu interaction theo implicit feedback:

- `interactions`: chỉ lưu bài user đã nghe (`listened = 1`).
- `training_pairs`: gồm positive và negative samples.

Lệnh sinh dữ liệu hiện tại:

```bash
python generate_synthetic_data.py --users 3000 --total-interactions 100000 --negative-ratio 1.0
```

Kết quả:

| Thành phần | Số lượng |
|---|---:|
| Users | 3000 |
| Songs | 8801 |
| Interactions đã nghe | 100000 |
| Training pairs | 200000 |
| Positive samples | 100000 |
| Negative samples | 100000 |

## 3.5. Chiến lược sinh positive/negative samples

Positive samples là các bài user được xem là đã nghe. Dữ liệu được sinh theo hướng cân bằng hơn giữa sở thích cố định và hành vi khám phá:

- phần lớn bài đã nghe vẫn bám theo thể loại yêu thích;
- một phần bài được chọn ngoài thể loại yêu thích để mô phỏng hành vi nghe khám phá;
- ngôn ngữ yêu thích được ưu tiên nhưng không ép tuyệt đối;
- độ tương đồng giữa sở thích âm thanh của user và audio features vẫn giữ vai trò quan trọng;
- nhiễu ngẫu nhiên được thêm vào để tránh dữ liệu quá sạch.

Điểm phù hợp khi sinh positive samples được tính theo hướng:

```text
affinity = 0.50 * audio_similarity
         + 0.30 * genre_match
         + 0.20 * language_match
         + noise
```

Negative samples là các bài user chưa nghe, được lấy từ phần còn lại của catalog. Negative samples giúp model học ranh giới giữa bài phù hợp và bài chưa phù hợp.

## 3.6. Hai mô hình được đề xuất

Đề tài sử dụng hai mô hình học máy trên dữ liệu bảng:

| Mô hình | Vai trò |
|---|---|
| RandomForestClassifier | baseline mạnh, dễ hiểu, ổn định |
| LightGBMClassifier | mô hình boosting hiệu quả, thường tốt trên dữ liệu bảng |

Cả hai nhận đầu vào là user features + song features, đầu ra là xác suất `listened = 1`.

## 3.7. RandomForestClassifier

RandomForestClassifier là mô hình ensemble dùng nhiều cây quyết định. Mỗi cây học trên một mẫu dữ liệu khác nhau, sau đó mô hình kết hợp kết quả biểu quyết/xác suất của nhiều cây.

Trong bài toán này, RandomForestClassifier học quan hệ giữa:

```text
user profile + song features -> listened
```

Ưu điểm:

- dễ hiểu;
- ít yêu cầu chuẩn hóa dữ liệu;
- ổn định;
- phù hợp làm baseline.

Nhược điểm:

- có thể nặng khi số cây lớn;
- inference chậm hơn boosting;
- kết quả hiện tại kém LightGBM.

## 3.8. LightGBMClassifier

LightGBMClassifier là mô hình gradient boosting trên cây quyết định, tối ưu cho dữ liệu bảng và dữ liệu lớn. Các cây được xây dựng tuần tự, cây sau tập trung sửa lỗi của cây trước.

Trong bài toán này, LightGBMClassifier dự đoán:

```text
match_score = P(listened = 1 | user, song)
```

Ưu điểm:

- train nhanh;
- hiệu quả trên dữ liệu bảng;
- học tốt quan hệ phi tuyến;
- kết quả hiện tại tốt nhất trong hai mô hình.

Nhược điểm:

- cần cài thư viện `lightgbm`;
- nhạy với tham số;
- khó giải thích hơn RandomForest nếu không phân tích feature importance.

## 3.9. Huấn luyện mô hình

### 3.9.1. Feature engineering

Feature gồm:

| Nhóm | Feature |
|---|---|
| Similarity numeric | preference_similarity, genre_match_score, language_match_score |
| User numeric | age, preferred_energy, preferred_valence, preferred_danceability, preferred_tempo, preferred_acousticness, preferred_instrumentalness, preferred_liveness, preferred_speechiness |
| Song numeric | duration_norm, energy, valence, danceability, tempo_norm, acousticness, instrumentalness, liveness, speechiness |
| Categorical | age_group, gender, language_preference, favorite_genres, favorite_detailed_genres, genre_top, genres_titles_text, genres_all_titles_text, language_code |

Đặc trưng số được truyền trực tiếp. Đặc trưng phân loại được mã hóa One-Hot bằng `OneHotEncoder(handle_unknown="ignore")`. Các feature `preference_similarity`, `genre_match_score`, `language_match_score` được dùng để giúp model nhận biết rõ hơn mức độ khớp giữa hồ sơ user và bài hát.

### 3.9.2. Train/test split

Không dùng random row split toàn cục. Hệ thống dùng per-user holdout:

- với mỗi user, positive samples được chia train/test;
- negative samples được chia train/test;
- train/test đều giữ phân phối theo từng user.

Cách chia này mô phỏng sát hơn bài toán thực tế: với một user đã có một phần lịch sử nghe, hệ thống cần dự đoán các bài phù hợp còn lại.

### 3.9.3. Dự đoán

Model trả về xác suất class 1:

```python
predict_proba(X)[:, 1]
```

Xác suất này được gọi là `match_score`.

## 3.10. Quy trình gợi ý Top-N

Quy trình gợi ý cho một user:

1. Lấy profile user từ bảng `user_profiles`.
2. Lấy danh sách bài hát từ bảng `songs`.
3. Lấy các bài user đã nghe từ `interactions`.
4. Loại bỏ các bài đã nghe khỏi candidate set.
5. Tạo feature cho từng cặp user-song còn lại.
6. Dùng model dự đoán `match_score`.
7. Sắp xếp giảm dần theo `match_score`.
8. Lấy Top-N bài hát làm gợi ý.

# CHƯƠNG 4. KẾT QUẢ THỰC NGHIỆM

## 4.1. Kết quả dữ liệu

Sau tiền xử lý, hệ thống có:

| Thành phần | Giá trị |
|---|---:|
| Số bài hát | 8801 |
| Số user mô phỏng | 3000 |
| Số interaction đã nghe | 100000 |
| Số training pairs | 200000 |

Phân phối label trong `training_pairs`:

| listened | Số lượng |
|---:|---:|
| 0 | 100000 |
| 1 | 100000 |

Như vậy dữ liệu huấn luyện cân bằng giữa positive và negative samples.

## 4.2. Kết quả LightGBMClassifier

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.9443 |
| Precision | 0.9247 |
| Recall | 0.9673 |
| F1 | 0.9455 |
| ROC-AUC | 0.9792 |
| Precision@10 | 0.7679 |
| Recall@10 | 0.9021 |
| NDCG@10 | 0.9753 |
| Users evaluated | 1840 |

Nhận xét: LightGBMClassifier đạt kết quả cao trên cả classification metrics và ranking metrics. ROC-AUC đạt 0.9792, cho thấy mô hình phân biệt tốt giữa bài đã nghe và chưa nghe. NDCG@10 đạt 0.9753, phản ánh chất lượng xếp hạng Top-N tốt, các bài phù hợp có xu hướng xuất hiện ở vị trí cao.

## 4.3. Kết quả RandomForestClassifier

| Metric | Giá trị |
|---|---:|
| Accuracy | 0.8649 |
| Precision | 0.8317 |
| Recall | 0.9149 |
| F1 | 0.8713 |
| ROC-AUC | 0.9557 |
| Precision@10 | 0.7528 |
| Recall@10 | 0.8886 |
| NDCG@10 | 0.9520 |
| Users evaluated | 1840 |

Nhận xét: RandomForestClassifier cải thiện rõ rệt sau khi dữ liệu synthetic được sinh lại theo hướng bám sát sở thích user hơn. Tuy vậy, mô hình vẫn thấp hơn LightGBMClassifier ở các chỉ số tổng hợp như F1, ROC-AUC và NDCG@10.

## 4.4. So sánh hai mô hình

| Model | F1 | ROC-AUC | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| LightGBMClassifier | 0.9455 | 0.9792 | 0.7679 | 0.9021 | 0.9753 |
| RandomForestClassifier | 0.8713 | 0.9557 | 0.7528 | 0.8886 | 0.9520 |

LightGBMClassifier tốt hơn RandomForestClassifier trên:

- F1;
- ROC-AUC;
- Precision@10;
- Recall@10;
- NDCG@10.

Do đó LightGBMClassifier phù hợp làm mô hình chính cho hệ gợi ý hiện tại, trong khi RandomForestClassifier đóng vai trò baseline để đối chiếu. Kết quả cao một phần đến từ dữ liệu synthetic có quy luật rõ giữa hồ sơ user và bài đã nghe; vì vậy khi trình bày cần nêu đây là đánh giá trên dữ liệu mô phỏng, chưa thay thế kiểm thử trên dữ liệu người dùng thật.

## 4.5. Triển khai web demo

Web demo Flask gồm các chức năng:

- đăng ký/đăng nhập bằng tài khoản thật;
- mật khẩu được hash trước khi lưu;
- nhập hồ sơ cá nhân và sở thích âm thanh;
- cập nhật profile bằng modal trong trang hồ sơ;
- duyệt danh sách bài hát;
- xem chi tiết bài hát;
- bấm `Nghe nhạc` để ghi nhận hành vi đã nghe;
- tự cập nhật một phần hồ sơ âm thanh của user theo bài vừa nghe;
- xem danh sách bài đã nghe;
- xem gợi ý cá nhân hóa theo `match_score`;
- xem dashboard admin với metrics classification/ranking.

Các trang chính:

| Route | Chức năng |
|---|---|
| `/login` | Đăng nhập |
| `/register` | Đăng ký |
| `/setup` | Nhập thông tin ban đầu |
| `/home` | Danh sách bài hát |
| `/recommendations` | Gợi ý cá nhân hóa |
| `/profile` | Hồ sơ và bài đã nghe |
| `/admin` | Dashboard dữ liệu và mô hình |

# CHƯƠNG 5. KẾT LUẬN

## 5.1. Kết quả đạt được

Đề tài đã xây dựng được hệ gợi ý nhạc dựa trên thông tin cá nhân bằng cách kết hợp dữ liệu bài hát từ FMA, hồ sơ user và dữ liệu nghe nhạc mô phỏng. Hệ thống sử dụng implicit feedback để học từ trạng thái bài đã nghe/chưa nghe và dự đoán xác suất phù hợp giữa user và bài hát.

Các kết quả chính:

- Tiền xử lý được dữ liệu FMA metadata và Echonest audio features.
- Lưu dữ liệu bài hát vào SQLite.
- Xây dựng hệ đăng ký/đăng nhập thật.
- Lưu hồ sơ user và sở thích âm thanh.
- Sinh được 3000 user mô phỏng và 100000 interaction đã nghe.
- Tạo được 200000 training pairs cân bằng positive/negative.
- Sinh dữ liệu theo hướng cân bằng giữa sở thích cố định và hành vi khám phá.
- Mô hình hóa bài toán dưới dạng binary classification với nhãn `listened`.
- Triển khai RandomForestClassifier và LightGBMClassifier.
- Đánh giá bằng Accuracy, Precision, Recall, F1, ROC-AUC, Precision@10, Recall@10, NDCG@10.
- Gợi ý Top-N bằng `match_score` trong khoảng 0–1.
- Xây dựng web demo Flask hoàn chỉnh.

## 5.2. Hạn chế

- Dữ liệu user và interaction hiện tại vẫn là mô phỏng, chưa phải hành vi thật.
- Negative samples được sinh nhân tạo nên chưa phản ánh đầy đủ việc user thật bỏ qua bài hát.
- Chưa có cơ chế ghi nhận hành vi thật như click, nghe hết bài, nghe lại, skip, favorite.
- Chưa có cơ chế cập nhật model online sau khi user sử dụng web.
- Chưa có phần giải thích chi tiết vì sao một bài được gợi ý.

## 5.3. Hướng phát triển

- Thu thập interaction thật từ web app.
- Mở rộng implicit feedback: click, play, skip, replay, favorite, add playlist.
- Tối ưu negative sampling theo hành vi thực tế.
- Thử các mô hình ranking chuyên biệt như LambdaMART hoặc learning-to-rank.
- Bổ sung feature importance và giải thích gợi ý.
- Cập nhật model định kỳ khi có thêm dữ liệu mới.
