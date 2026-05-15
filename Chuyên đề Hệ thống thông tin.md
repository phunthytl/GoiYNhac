# **CHƯƠNG 1\. GIỚI THIỆU CHUNG** {#chương-1.-giới-thiệu-chung}

## **1.1. Bối cảnh đề tài** {#1.1.-bối-cảnh-đề-tài}

Trong thời đại số, các nền tảng nghe nhạc trực tuyến như Spotify, Apple Music, YouTube Music hoặc SoundCloud cung cấp cho người dùng một kho nhạc rất lớn. Số lượng bài hát ngày càng tăng giúp người dùng có nhiều lựa chọn hơn, nhưng đồng thời cũng tạo ra vấn đề quá tải thông tin. Người dùng khó tự tìm được bài hát phù hợp với sở thích cá nhân, tâm trạng, ngôn ngữ yêu thích hoặc thói quen nghe nhạc của mình.

Hệ gợi ý nhạc được xây dựng nhằm hỗ trợ người dùng khám phá bài hát phù hợp hơn. Thay vì chỉ hiển thị danh sách bài hát phổ biến chung cho tất cả mọi người, hệ thống có thể tận dụng thông tin cá nhân và đặc trưng âm thanh của bài hát để đưa ra danh sách gợi ý mang tính cá nhân hóa.

Đề tài “Hệ gợi ý nhạc dựa trên thông tin cá nhân” tập trung xây dựng một hệ thống gợi ý sử dụng dữ liệu metadata và audio features từ bộ dữ liệu FMA, kết hợp với hồ sơ người dùng và dữ liệu tương tác được mô phỏng. Hệ thống huấn luyện mô hình học máy để dự đoán mức độ phù hợp giữa người dùng và bài hát, sau đó xếp hạng các bài hát theo điểm dự đoán để tạo danh sách gợi ý Top-N.

## **1.2. Mục tiêu đề tài** {#1.2.-mục-tiêu-đề-tài}

Đề tài hướng đến mục tiêu xây dựng một hệ thống gợi ý nhạc cá nhân hóa có khả năng sử dụng thông tin hồ sơ người dùng, sở thích âm thanh và đặc trưng bài hát để dự đoán mức độ phù hợp giữa người dùng và từng bài hát. Từ đó, hệ thống xếp hạng và đề xuất danh sách bài hát phù hợp nhất cho từng người dùng.

Các mục tiêu cụ thể gồm:

1. Tìm hiểu cơ sở lý thuyết về hệ gợi ý, đặc biệt là bài toán gợi ý nhạc cá nhân hóa và gợi ý dựa trên hồ sơ người dùng.

2. Khai thác bộ dữ liệu FMA metadata, tiền xử lý thông tin bài hát và đặc trưng âm thanh, sau đó lưu trữ dữ liệu đã xử lý trong cơ sở dữ liệu SQLite để phục vụ web app và mô hình gợi ý.

3. Xây dựng dữ liệu interaction/rating mô phỏng có kiểm soát do FMA không cung cấp đầy đủ thông tin đánh giá cá nhân của người dùng.

4. Thiết kế bộ đặc trưng kết hợp giữa hồ sơ người dùng, metadata bài hát, đặc trưng âm thanh và các đặc trưng so khớp giữa sở thích người dùng với bài hát.

5. Triển khai và so sánh hai mô hình học máy trên dữ liệu bảng, gồm RandomForestRegressor và LightGBMRegressor, cho bài toán dự đoán rating/preference score.

6. Đánh giá chất lượng mô hình bằng các chỉ số hồi quy và xếp hạng như RMSE, MAE, Precision@10, Recall@10 và NDCG@10.

7. Xây dựng web demo Flask có đăng ký/đăng nhập tài khoản, lưu hồ sơ cá nhân, cập nhật sở thích nghe nhạc và hiển thị danh sách gợi ý Top-N cho từng người dùng.

## **1.3. Bài toán nghiên cứu** {#1.3.-bài-toán-nghiên-cứu}

Bài toán được đặt ra như sau:

\- Với một người dùng có thông tin cá nhân và sở thích âm nhạc ban đầu, hệ thống cần dự đoán mức độ phù hợp của từng bài hát đối với người dùng đó, sau đó xếp hạng và đề xuất các bài hát phù hợp nhất.

Đầu vào của bài toán gồm:

- thông tin người dùng,

- sở thích thể loại,

- ngôn ngữ nhạc yêu thích,

- sở thích về năng lượng, nhịp độ, độ vui tươi, độ phổ biến,

- thông tin bài hát,

- đặc trưng âm thanh của bài hát.

Đầu ra của bài toán là:

\- predicted\_rating(user, track)

hoặc có thể hiểu là:

\- preference\_score(user, track)

Sau khi có điểm dự đoán, hệ thống sắp xếp các bài hát theo điểm giảm dần và lấy Top-N bài hát làm danh sách gợi ý.

# **CHƯƠNG 2\. CƠ SỞ LÝ THUYẾT** {#chương-2.-cơ-sở-lý-thuyết}

## **2.1. Khái niệm hệ gợi ý** {#2.1.-khái-niệm-hệ-gợi-ý}

Hệ gợi ý là một hệ thống có khả năng đề xuất các đối tượng phù hợp với từng người dùng dựa trên dữ liệu đã thu thập được. Các đối tượng được gợi ý có thể là phim, bài hát, sản phẩm, khóa học, bài viết, video hoặc bất kỳ item nào trong một hệ thống thông tin.

Trong bài toán gợi ý nhạc:

- Người dùng \= listener / user

- Đối tượng gợi ý \= track / song / bài hát

Mục tiêu của hệ gợi ý không chỉ là tìm bài hát phổ biến nhất, mà là tìm bài hát phù hợp nhất với từng người dùng cụ thể. Hai người dùng khác nhau có thể nhận hai danh sách gợi ý khác nhau dù truy cập cùng một hệ thống.

Một hệ gợi ý thường cần giải quyết ba câu hỏi chính:

1. Người dùng là ai?  
2. Item có đặc điểm gì?  
3. Người dùng có khả năng thích item đó đến mức nào?

Với hệ gợi ý nhạc dựa trên thông tin cá nhân, hệ thống đặc biệt quan tâm đến hồ sơ người dùng và sở thích nghe nhạc ban đầu, chẳng hạn thể loại yêu thích, ngôn ngữ yêu thích, độ sôi động mong muốn hoặc mức độ phổ biến của bài hát.

## **2.2. Vai trò của hệ gợi ý trong ứng dụng nghe nhạc** {#2.2.-vai-trò-của-hệ-gợi-ý-trong-ứng-dụng-nghe-nhạc}

Trong ứng dụng nghe nhạc, hệ gợi ý có nhiều vai trò quan trọng:

- Giúp người dùng khám phá bài hát mới.  
- Giảm thời gian tìm kiếm thủ công.  
- Cá nhân hóa trải nghiệm nghe nhạc.  
- Tăng mức độ tương tác của người dùng với hệ thống.  
- Hỗ trợ xây dựng playlist tự động.  
- Giữ chân người dùng nhờ nội dung phù hợp hơn.

Nếu không có hệ gợi ý, người dùng thường chỉ thấy các bảng xếp hạng chung như bài hát phổ biến hoặc bài hát mới phát hành. Những danh sách này không phản ánh đầy đủ sở thích cá nhân của từng người.

Ví dụ, một người thích nhạc Electronic sôi động và một người thích Classical nhẹ nhàng không nên nhận cùng một danh sách gợi ý. Hệ gợi ý giúp hệ thống phân biệt và cá nhân hóa kết quả cho từng người dùng.

## **2.3. Các thành phần cơ bản của hệ gợi ý** {#2.3.-các-thành-phần-cơ-bản-của-hệ-gợi-ý}

Một hệ gợi ý thường gồm ba thành phần dữ liệu chính:

### ***2.3.1. User*** {#2.3.1.-user}

User là người sử dụng hệ thống. Thông tin user có thể bao gồm:

- mã người dùng,  
- tuổi,  
- giới tính,  
- ngôn ngữ yêu thích,  
- thể loại yêu thích,  
- lịch sử nghe nhạc,  
- lịch sử đánh giá,  
- hành vi tương tác.

Trong đề tài này, user được mô tả bằng các thông tin cá nhân và sở thích âm thanh như:

- age  
- gender  
- favorite\_genres  
- language\_preference  
- preferred\_energy  
- preferred\_valence  
- preferred\_danceability  
- preferred\_tempo  
- preferred\_popularity

### ***2.3.2. Item*** {#2.3.2.-item}

Item là đối tượng cần gợi ý. Trong bài toán này, item là bài hát.

Một bài hát có thể được mô tả bởi:

- tên bài hát,  
- nghệ sĩ,  
- album,  
- thể loại,  
- ngôn ngữ,  
- thời lượng,  
- lượt nghe,  
- lượt yêu thích,  
- đặc trưng âm thanh.

Trong project, item được lấy từ FMA metadata, được tiền xử lý và lưu trực tiếp vào cơ sở dữ liệu SQLite `data/musics.db`, bảng `songs`.

### ***2.3.3. Interaction*** {#2.3.3.-interaction}

Interaction là tương tác giữa user và item. Trong hệ gợi ý nhạc, interaction có thể là:

- nghe bài hát,  
- bỏ qua bài hát,  
- thích bài hát,  
- thêm vào playlist,  
- đánh giá sao,  
- nghe lại nhiều lần.

## **2.4. Các hướng tiếp cận phổ biến trong hệ gợi ý** {#2.4.-các-hướng-tiếp-cận-phổ-biến-trong-hệ-gợi-ý}

### ***2.4.1. Popularity-Based Recommendation*** {#2.4.1.-popularity-based-recommendation}

Popularity-Based Recommendation là cách gợi ý đơn giản nhất. Hệ thống đề xuất các item phổ biến nhất cho tất cả người dùng.

Ví dụ: Gợi ý top 10 bài hát có lượt nghe cao nhất.

Ưu điểm:

- Dễ triển khai.  
- Không cần thông tin cá nhân.  
- Phù hợp khi chưa có dữ liệu người dùng.

Nhược điểm:

- Không cá nhân hóa.  
- Người dùng khác nhau nhận cùng một kết quả.  
- Không phản ánh gu âm nhạc riêng.

### ***2.4.2. Content-Based Filtering*** {#2.4.2.-content-based-filtering}

Content-Based Filtering dựa trên đặc trưng của item. Trong bài toán nhạc, đặc trưng item có thể gồm:

- genre,  
- artist,  
- language,  
- duration,  
- energy,  
- valence,  
- danceability,  
- tempo,  
- popularity.

Hệ thống sẽ gợi ý các bài hát có đặc trưng giống với sở thích của người dùng.

Ví dụ: Nếu user thích Electronic và bài hát có energy cao, hệ thống ưu tiên các bài Electronic có energy tương tự.

Ưu điểm:

- Phù hợp với user mới nếu người dùng nhập sở thích ban đầu.  
- Có thể giải thích được vì sao một bài hát được gợi ý.  
- Không phụ thuộc hoàn toàn vào hành vi của cộng đồng.

Nhược điểm:

- Dễ bị giới hạn trong vùng sở thích cũ.  
- Khó gợi ý các bài hát khác gu nhưng người dùng có thể thích.  
- Chất lượng phụ thuộc vào đặc trưng item.

### ***2.4.3. Collaborative Filtering*** {#2.4.3.-collaborative-filtering}

Collaborative Filtering dựa trên hành vi của cộng đồng người dùng. Ý tưởng chính là những người dùng có hành vi giống nhau trong quá khứ có thể sẽ thích các item giống nhau trong tương lai.

Ví dụ: user A và user B cùng thích nhiều bài Pop, user B thích thêm bài X.

→ Hệ thống có thể gợi ý bài X cho User A.

Ưu điểm:

- Khai thác được hành vi cộng đồng.  
- Có thể gợi ý các bài hát ngoài sở thích khai báo ban đầu.  
- Không cần hiểu sâu nội dung bài hát.

Nhược điểm:

- Khó xử lý user mới chưa có lịch sử tương tác.  
- Khó xử lý bài hát mới chưa có tương tác.  
- Cần dữ liệu tương tác đủ lớn.  
- Dữ liệu thường thưa vì mỗi user chỉ nghe một phần rất nhỏ trong kho nhạc.

### ***2.4.4. Profile-Based Recommendation*** {#2.4.4.-profile-based-recommendation}

Profile-Based Recommendation sử dụng hồ sơ cá nhân của người dùng như:

- tuổi,  
- giới tính,  
- thể loại yêu thích,  
- ngôn ngữ yêu thích,  
- sở thích âm thanh.

Cách tiếp cận này phù hợp với bài toán của đề tài vì hệ thống cần gợi ý nhạc dựa trên thông tin cá nhân ban đầu.

Ưu điểm:

- Hữu ích khi user chưa có nhiều lịch sử nghe nhạc.  
- Dễ kết hợp với form nhập sở thích.  
- Có thể giải thích gợi ý dựa trên profile.

Nhược điểm:

- Nếu thông tin user nhập ít hoặc không chính xác, chất lượng gợi ý giảm.  
- Profile ban đầu có thể chưa phản ánh đầy đủ gu nghe nhạc thực tế.  
- Cần cập nhật profile theo hành vi thực tế nếu hệ thống triển khai lâu dài.

## **2.5. Bài toán cold-start** {#2.5.-bài-toán-cold-start}

Cold-start là một trong những vấn đề quan trọng nhất của hệ gợi ý.

Có hai loại cold-start phổ biến:

### ***2.5.1. User cold-start*** {#2.5.1.-user-cold-start}

User cold-start xảy ra khi người dùng mới chưa có lịch sử tương tác. Khi đó, các phương pháp dựa vào hành vi như Collaborative Filtering khó đưa ra gợi ý chính xác.

Cách xử lý:

- yêu cầu user nhập sở thích ban đầu,  
- dùng thông tin cá nhân,  
- dùng thể loại yêu thích,  
- dùng bài hát phổ biến trong nhóm sở thích.

### ***2.5.2. Item cold-start*** {#2.5.2.-item-cold-start}

Item cold-start xảy ra khi bài hát mới chưa có lượt nghe hoặc rating. Khi đó, hệ thống khó biết bài hát phù hợp với ai nếu chỉ dựa trên tương tác.

Cách xử lý:

- dùng metadata của bài hát,  
- dùng genre,  
- dùng audio features,  
- dùng artist/album/language.

## **2.6. Đặc trưng âm thanh trong gợi ý nhạc** {#2.6.-đặc-trưng-âm-thanh-trong-gợi-ý-nhạc}

Âm nhạc có nhiều đặc trưng khác với phim hoặc sản phẩm thương mại. Ngoài thể loại và nghệ sĩ, bài hát còn có các đặc điểm âm thanh như năng lượng, nhịp độ, sắc thái cảm xúc và độ phù hợp để nhảy.

Trong project, các đặc trưng âm thanh được biểu diễn bằng các chỉ số:

1. Energy: thể hiện mức độ mạnh, sôi động của bài nhạc. Energy thấp: nhẹ, êm, thư giãn. Energy cao: mạnh, dồn dập, sôi động.  
2. Valence: thể hiện sắc thái cảm xúc tích cực của bài nhạc. Valence thấp: buồn, tối, căng thẳng. Valence cao: vui, tươi, sáng.  
3. Danceability: thể hiện mức độ phù hợp để nhún nhảy hoặc bắt nhịp. Danceability cao: nhịp rõ, đều, dễ bắt beat.  
4. Tempo: thể hiện tốc độ cảm nhận của bài hát. Tempo thấp: chậm, nhẹ, thư giãn. Tempo cao: nhanh, sôi động.  
5. Popularity: thể hiện mức độ phổ biến của bài hát, thường liên quan đến lượt nghe, lượt thích hoặc mức độ quan tâm. Popularity thấp: bài ít nổi, phù hợp khám phá. Popularity cao: bài phổ biến hơn.

## **2.7.  Bài toán Regression trong hệ gợi ý** {#2.7.-bài-toán-regression-trong-hệ-gợi-ý}

Trong đề tài này, bài toán gợi ý được đưa về bài toán hồi quy:

- Input: user features \+ song features \+ matching features  
- Output: rating / preference score

Mô hình dự đoán một giá trị liên tục thể hiện mức độ phù hợp giữa người dùng và bài hát. Sau đó, hệ thống sắp xếp bài hát theo điểm dự đoán để đưa ra Top-N recommendation.

Cách tiếp cận này phù hợp khi dữ liệu có rating hoặc có thể quy đổi tương tác thành điểm số.

## **2.8. Các chỉ số đánh giá hệ gợi ý** {#2.8.-các-chỉ-số-đánh-giá-hệ-gợi-ý}

### ***2.8.1. RMSE*** {#2.8.1.-rmse}

RMSE là căn bậc hai của sai số bình phương trung bình.

RMSE \= sqrt(mean((y\_true \- y\_pred)^2))

Ý nghĩa:

* Càng thấp càng tốt.  
* Phạt mạnh các dự đoán sai lệch lớn.  
* Phù hợp để đánh giá mô hình dự đoán rating.

### ***2.8.2. MAE*** {#2.8.2.-mae}

MAE là sai số tuyệt đối trung bình.

MAE \= mean(abs(y\_true \- y\_pred))

Ý nghĩa:

* Càng thấp càng tốt.  
* Dễ hiểu hơn RMSE.  
* Cho biết trung bình mô hình dự đoán lệch bao nhiêu điểm rating.

### ***2.8.3. Precision@k*** {#2.8.3.-precision@k}

Precision@k đo trong k bài hát được gợi ý đầu tiên, có bao nhiêu bài thực sự phù hợp.

Precision@k \= số bài relevant trong Top k/ k

Ý nghĩa:

\- Càng cao càng tốt.

\- Phản ánh độ chính xác của danh sách gợi ý đầu tiên.

### ***2.8.4. Recall@k*** {#2.8.4.-recall@k}

Recall@k đo trong tất cả các bài phù hợp với user, hệ thống tìm lại được bao nhiêu bài trong Top k.

Recall@k \= số bài relevant trong Top k/ tổng số bài relevant

Ý nghĩa:

\- Càng cao càng tốt.

\- Phản ánh khả năng bao phủ các bài hát phù hợp.

### ***2.8.5. NDCG@k*** {#2.8.5.-ndcg@k}

NDCG@k  đánh giá chất lượng thứ hạng của Top k. Nếu bài phù hợp được xếp ở vị trí cao hơn, điểm NDCG sẽ cao hơn.

Ý nghĩa:

\- Càng gần 1 càng tốt.

\- Phù hợp với hệ gợi ý vì thứ tự hiển thị ảnh hưởng trực tiếp đến trải nghiệm người dùng.

## **2.9. Kết luận** {#2.9.-kết-luận}

Từ các cơ sở trên, đề tài chọn hướng tiếp cận dựa trên profile người dùng kết hợp đặc trưng bài hát. Bài toán được mô hình hóa thành bài toán hồi quy dự đoán rating/preference score, sau đó chuyển thành bài toán xếp hạng Top-N. Cách tiếp cận này phù hợp với bài toán gợi ý nhạc dựa trên thông tin cá nhân, đặc biệt trong bối cảnh dữ liệu user thực tế chưa có sẵn.

# 

# **CHƯƠNG 3\. PHƯƠNG PHÁP ĐỀ XUẤT** {#chương-3.-phương-pháp-đề-xuất}

## **3.1. Tổng quan phương pháp** {#3.1.-tổng-quan-phương-pháp}

Phương pháp đề xuất gồm các bước chính:

1. Thu thập và đọc dữ liệu FMA metadata.  
2. Tiền xử lý dữ liệu bài hát.  
3. Trích xuất và tạo đặc trưng âm thanh.  
4. Sinh dữ liệu hồ sơ người dùng.  
5. Sinh dữ liệu tương tác/rating mô phỏng.  
6. Tạo tập dữ liệu huấn luyện dạng user-track pairs.  
7. Sử dụng hai mô hình học máy để so sánh.  
8. Huấn luyện mô hình học máy dự đoán rating.  
9. Đánh giá mô hình bằng metrics dự đoán rating và ranking.  
10. Triển khai web demo để minh họa hệ thống gợi ý.

## **3.2. Dữ liệu sử dụng** {#3.2.-dữ-liệu-sử-dụng}

Đề tài sử dụng bộ dữ liệu FMA \- Free Music Archive. Gồm các file chính:

- tracks.csv: chứa thông tin bài hát, thông tin nghệ sĩ, thông tin album.  
- features.csv: chứa các đặc trưng âm thanh được trích xuất từ tín hiệu âm nhạc của từng bài hát  
- genres.csv: chứa thông tin về hệ thống thể loại nhạc trong FMA  
- echonest.csv: chứa metadata và đặc trưng bổ sung từ Echo Nest

Do FMA không cung cấp đầy đủ dữ liệu cá nhân và rating người dùng, project tạo thêm dữ liệu người dùng và interaction mô phỏng.

## **3.3. Tiền xử lý dữ liệu**  {#3.3.-tiền-xử-lý-dữ-liệu}

Quá trình tiền xử lý được thực hiện theo các bước tuần tự nhằm chuyển đổi dữ liệu thô thành tập dữ liệu hoàn chỉnh cho hệ gợi ý.

Bước 1: Thu thập và hợp nhất dữ liệu  
Dữ liệu được lấy từ bộ dữ liệu FMA kể trên, kết hợp lại dựa trên định danh bài hát để tạo thành một bảng dữ liệu thống nhất, trong đó mỗi dòng tương ứng với một bài hát.

Bước 2: Chuẩn hóa và làm sạch metadata  
Các thuộc tính quan trọng như tiêu đề, nghệ sĩ, album, thể loại, ngôn ngữ, thời lượng và các chỉ số tương tác được chọn lọc và chuẩn hóa. Dữ liệu thể loại được chuyển từ dạng mã sang tên dễ hiểu, đồng thời phân tách giữa thể loại chính và thể loại chi tiết. Các bản ghi thiếu thông tin quan trọng được loại bỏ hoặc xử lý bằng cách điền giá trị phù hợp.

Bước 3: Xử lý đặc trưng âm thanh  
Các đặc trưng âm thanh được gom nhóm theo từng loại và tính toán các giá trị đại diện như trung bình và độ biến thiên. Cách làm này giúp giảm số chiều dữ liệu nhưng vẫn giữ được các đặc điểm quan trọng của tín hiệu âm thanh.

Bước 4: Chuẩn hóa dữ liệu và xây dựng đặc trưng tổng hợp  
Toàn bộ các đặc trưng số được đưa về cùng thang đo nhằm đảm bảo tính nhất quán. Từ đó, xây dựng các đặc trưng tổng hợp như mức năng lượng, cảm xúc, khả năng nhảy, nhịp độ tương đối và độ phổ biến. Các đặc trưng này phản ánh đặc tính nội dung và mức độ quan tâm của người dùng đối với bài hát.

Bước 5: Giảm kích thước dữ liệu (nếu cần)  
Để phục vụ mục đích thử nghiệm, tập dữ liệu có thể được giới hạn bằng cách ưu tiên các bài hát phổ biến và thuộc các tập con nhỏ hơn, giúp giảm chi phí tính toán nhưng vẫn đảm bảo tính đa dạng.

Bước 6: Lưu trữ dữ liệu đầu ra  
Sau khi hoàn tất các bước xử lý, dữ liệu bài hát được ghi trực tiếp vào cơ sở dữ liệu SQLite `data/musics.db`, bảng `songs`. Cách lưu trữ này giúp web app, bước sinh dữ liệu interaction và bước gợi ý sử dụng chung một nguồn dữ liệu thay vì phải đọc lại file CSV trung gian.

## **3.4. Sinh hồ sơ người dùng và interaction** {#3.4.-sinh-hồ-sơ-người-dùng-và-interaction}

Do dữ liệu FMA không cung cấp thông tin người dùng và tương tác, hệ thống tiến hành sinh dữ liệu giả lập nhằm mô phỏng hành vi nghe nhạc phục vụ huấn luyện mô hình. Quá trình này được thiết kế theo các bước sau:

Bước 1: Khởi tạo hồ sơ người dùng  
Một tập người dùng được tạo ra với các thuộc tính cơ bản như tuổi, nhóm tuổi, giới tính và ngôn ngữ ưa thích. Đồng thời, mỗi người dùng được gán một tập thể loại yêu thích và các sở thích âm nhạc dưới dạng số như mức năng lượng, cảm xúc, khả năng nhảy, nhịp độ và độ phổ biến mong muốn. Các giá trị này được sinh ngẫu nhiên nhưng có kiểm soát để đảm bảo phân phối hợp lý.

Bước 2: Xác định số lượng tương tác cho mỗi người dùng  
Số lượng bài hát mà mỗi người dùng tương tác không cố định mà được lấy theo một phân phối giảm dần. Phần lớn người dùng có ít tương tác, trong khi một tỷ lệ nhỏ có số lượng tương tác lớn, giúp mô phỏng gần hơn với dữ liệu thực tế.

Bước 3: Lấy mẫu bài hát cho từng người dùng  
Đối với mỗi người dùng, các bài hát được chọn theo chiến lược kết hợp:

- Phần lớn (khoảng 70%) được lấy từ các thể loại yêu thích  
- Phần còn lại được chọn ngẫu nhiên từ toàn bộ tập dữ liệu

Cách tiếp cận này giúp tạo ra cả tương tác tích cực (phù hợp sở thích) và tương tác trung tính/tiêu cực, cần thiết cho việc học mô hình.

Bước 4: Tính điểm phù hợp và sinh rating  
Mỗi cặp người dùng \- bài hát được gán một điểm phù hợp dựa trên:

- Mức độ trùng khớp thể loại và ngôn ngữ  
- Khoảng cách giữa sở thích người dùng và đặc trưng bài hát (energy, valence, ...)  
- Độ phổ biến của bài hát

Rating được sinh theo thang điểm 1–5 với phân phối có kiểm soát, trong đó đa số rating nằm ở mức trung bình/khá nhưng vẫn có rating thấp 1–2 sao để dữ liệu cân bằng và thực tế hơn. Độ khớp giữa hồ sơ người dùng và bài hát làm rating tăng hoặc giảm, đồng thời nhiễu ngẫu nhiên được thêm vào để mô phỏng sự không chắc chắn trong hành vi nghe nhạc. Hệ thống không lưu thêm nhãn `liked` vì nhãn này có thể suy ra trực tiếp từ rating nếu cần đánh giá ranking.

Bước 5: Tạo dữ liệu huấn luyện mở rộng  
 Ngoài bảng tương tác cơ bản, hệ thống xây dựng thêm một tập dữ liệu kết hợp đầy đủ thông tin người dùng, bài hát và các đặc trưng so khớp (như chênh lệch energy, valence,...). Tập dữ liệu này phục vụ cho các mô hình học có giám sát.

Bước 6: Lưu trữ dữ liệu đầu ra  
Kết quả cuối cùng được lưu vào SQLite gồm:

- `interactions`: tương tác người dùng – bài hát, gồm `user_id`, `track_id`, `rating`.  
- `training_pairs`: dữ liệu huấn luyện mở rộng, kết hợp thông tin user, song và matching features.

Hồ sơ người dùng thật của web app được lưu riêng trong bảng `user_profiles` và liên kết với bảng `accounts` để phục vụ đăng nhập/đăng ký.

## **3.5. Hai mô hình được đề xuất** {#3.5.-hai-mô-hình-được-đề-xuất}

Đề tài lựa chọn hai mô hình học máy trên dữ liệu bảng để so sánh:

- RandomForestRegressor  
- LightGBMRegressor

Cả hai mô hình đều nhận đầu vào là bộ đặc trưng kết hợp giữa người dùng, bài hát và matching features. Đầu ra là rating dự đoán.

## **3.6. Mô hình RandomForestRegressor** {#3.6.-mô-hình-randomforestregressor}

### ***3.6.1. Giới thiệu*** {#3.6.1.-giới-thiệu}

RandomForestRegressor là mô hình học máy thuộc nhóm ensemble learning, sử dụng nhiều cây quyết định để dự đoán giá trị liên tục. Mỗi cây được huấn luyện trên một mẫu dữ liệu khác nhau, sau đó kết quả dự đoán cuối cùng là trung bình dự đoán của nhiều cây.

Random Forest dựa trên kỹ thuật bagging: Bootstrap Aggregating. Tức là mô hình tạo nhiều tập con dữ liệu bằng cách lấy mẫu có hoàn lại, huấn luyện nhiều cây quyết định độc lập và kết hợp kết quả.

### ***3.6.2. Cách hoạt động*** {#3.6.2.-cách-hoạt-động}

Quy trình tổng quát:

1. Tạo nhiều tập dữ liệu con từ tập huấn luyện ban đầu.  
2. Huấn luyện một Decision Tree trên mỗi tập con.  
3. Mỗi cây đưa ra một dự đoán rating.  
4. Lấy trung bình dự đoán của tất cả cây làm kết quả cuối cùng.

Công thức khái quát:

y\_pred \= average(tree\_1(x), tree\_2(x), ..., tree\_n(x))

### ***3.6.3. Vai trò trong bài toán gợi ý nhạc*** {#3.6.3.-vai-trò-trong-bài-toán-gợi-ý-nhạc}

Trong hệ gợi ý nhạc, RandomForestRegressor có thể học quan hệ phi tuyến giữa:

user profile \+ song features \+ matching features

và: rating / preference score

Ví dụ, mô hình có thể học rằng:

- người dùng thích Electronic có xu hướng rating cao các bài Electronic,  
- nếu energy\_diff nhỏ thì rating thường cao,  
- nếu genre\_match \= 1 và language\_match \= 1 thì bài hát có khả năng phù hợp hơn.

### ***3.6.4. Ưu điểm*** {#3.6.4.-ưu-điểm}

- Dễ hiểu hơn so với nhiều mô hình boosting phức tạp.  
- Hoạt động tốt trên dữ liệu bảng.  
- Ít yêu cầu chuẩn hóa dữ liệu.  
- Giảm overfitting so với một cây quyết định đơn lẻ.  
- Có thể phân tích feature importance.  
- Phù hợp làm baseline mạnh cho bài toán regression.

### ***3.6.5. Nhược điểm*** {#3.6.5.-nhược-điểm}

- Model có thể khá nặng nếu số lượng cây lớn.  
- Dự đoán chậm hơn một số mô hình boosting tối ưu.  
- Có thể kém chính xác hơn LightGBM trên dữ liệu bảng đã được feature engineering tốt.  
- Khó học các pattern rất tinh vi nếu tham số chưa được tối ưu.

## **3.7. Mô hình LightGBMRegressor** {#3.7.-mô-hình-lightgbmregressor}

### ***3.7.1. Giới thiệu*** {#3.7.1.-giới-thiệu}

LightGBMRegressor là mô hình gradient boosting do Microsoft phát triển, tối ưu cho dữ liệu bảng và dữ liệu lớn. LightGBM xây dựng nhiều cây quyết định theo cách tuần tự, trong đó mỗi cây sau tập trung sửa lỗi của các cây trước.

LightGBM thuộc nhóm: Gradient Boosting Decision Trees. Khác với Random Forest, các cây trong LightGBM không độc lập hoàn toàn. Mỗi cây mới được huấn luyện dựa trên phần sai số còn lại của mô hình hiện tại.

### ***3.7.2. Cách hoạt động*** {#3.7.2.-cách-hoạt-động}

Quy trình tổng quát:

1. Bắt đầu với một dự đoán ban đầu.  
2. Tính sai số giữa dự đoán và rating thật.  
3. Huấn luyện cây mới để giảm sai số đó.  
4. Cộng cây mới vào mô hình với một hệ số learning rate.  
5. Lặp lại nhiều vòng.

Công thức khái quát:

F\_m(x) \= F\_{m-1}(x) \+ learning\_rate \* tree\_m(x)

Trong đó:

- F\_m(x) là mô hình sau vòng thứ m,  
- tree\_m(x) là cây mới học phần lỗi còn lại,  
- learning\_rate điều chỉnh mức đóng góp của cây mới.

### ***3.7.3. Vai trò trong bài toán gợi ý nhạc*** {#3.7.3.-vai-trò-trong-bài-toán-gợi-ý-nhạc}

LightGBMRegressor được dùng để dự đoán rating giữa người dùng và bài hát:

predicted\_rating \= model(user\_features, song\_features, matching\_features)

Sau đó hệ thống xếp hạng các bài hát theo \`predicted\_rating\` để tạo danh sách gợi ý.

### ***3.7.4. Ưu điểm*** {#3.7.4.-ưu-điểm}

- Rất mạnh trên dữ liệu bảng.  
- Train nhanh và hiệu quả với số lượng dữ liệu lớn.  
- Học tốt các quan hệ phi tuyến giữa đặc trưng.  
- Có khả năng xử lý nhiều feature numeric/categorical sau khi encoding.  
- Thường đạt kết quả tốt hơn các mô hình truyền thống nếu dữ liệu được feature engineering tốt.

### ***3.7.5. Nhược điểm*** {#3.7.5.-nhược-điểm}

- Cần cài thêm thư viện lightgbm.  
- Nhạy hơn với tham số so với Random Forest.  
- Có thể overfit nếu dữ liệu nhỏ hoặc tham số không phù hợp.  
- Khó giải thích hơn Random Forest nếu không phân tích feature importance.

## **3.8. Huấn luyện mô hình** {#3.8.-huấn-luyện-mô-hình}

Sau khi chuẩn bị dữ liệu và đặc trưng, hệ thống tiến hành huấn luyện mô hình dự đoán rating (thang điểm 1–5). Bài toán được xây dựng dưới dạng hồi quy.

Bước 1: Chuẩn bị dữ liệu

Tập dữ liệu được chia thành: 80% huấn luyện và 20% kiểm tra

Bước 2: Tiền xử lý đặc trưng

- Đặc trưng số được chuẩn hóa về cùng thang đo

- Đặc trưng phân loại được mã hóa one-hot

Hai bước này được tích hợp trong một pipeline để đảm bảo xử lý nhất quán giữa train và test.

Bước 3: Mô hình LightGBM

Mô hình LightGBM sử dụng phương pháp gradient boosting trên cây quyết định, trong đó các cây được huấn luyện tuần tự để sửa lỗi của các cây trước.

Các tham số chính:

- n\_estimators \= 450: số lượng cây trong mô hình. Giá trị lớn giúp mô hình học tốt hơn nhưng tăng thời gian huấn luyện và nguy cơ overfitting.  
- learning\_rate \= 0.045: tốc độ học của mỗi cây. Giá trị nhỏ giúp mô hình học chậm nhưng ổn định hơn, thường đi kèm với số lượng cây lớn.

- num\_leaves \= 31: số lá tối đa của mỗi cây. Tham số này quyết định độ phức tạp của cây (càng lớn → mô hình càng dễ overfit).

- subsample \= 0.85: tỷ lệ lấy mẫu dữ liệu cho mỗi cây. Giúp giảm overfitting bằng cách không dùng toàn bộ dữ liệu mỗi lần học.

- colsample\_bytree \= 0.85: tỷ lệ chọn ngẫu nhiên các đặc trưng khi xây dựng mỗi cây. Giúp tăng tính đa dạng giữa các cây.

- objective \= "regression": xác định bài toán là hồi quy.

- random\_state: đảm bảo khả năng tái lập kết quả.

Bước 4: Mô hình Random Forest

Random Forest là mô hình bagging, trong đó nhiều cây quyết định được huấn luyện độc lập trên các mẫu dữ liệu khác nhau.

Các tham số chính:

- n\_estimators \= 220: số lượng cây trong rừng. Nhiều cây giúp kết quả ổn định hơn nhưng tăng chi phí tính toán.

- max\_depth \= 18: độ sâu tối đa của mỗi cây. Giới hạn này giúp kiểm soát overfitting.

- min\_samples\_leaf \= 3: số mẫu tối thiểu ở mỗi lá. Giá trị lớn hơn giúp mô hình mượt hơn và giảm nhiễu.

- max\_features \= sqrt: số lượng đặc trưng được xem xét tại mỗi lần chia node (lấy căn bậc hai tổng số feature). Điều này giúp tăng tính đa dạng giữa các cây.

- n\_jobs \= \-1: sử dụng toàn bộ CPU để tăng tốc huấn luyện.

- random\_state: đảm bảo kết quả có thể tái lập.

Bước 5: Dự đoán và đánh giá

Sau khi huấn luyện, mô hình dự đoán rating trên tập kiểm tra và giới hạn giá trị trong khoảng \[1, 5\].

Hai nhóm chỉ số được sử dụng:

- Sai số hồi quy: RMSE: đo độ lệch tổng thể, MAE: đo sai số trung bình tuyệt đối

- Chỉ số xếp hạng: Precision@K, Recall@K, NDCG@K

Các chỉ số này được tính theo từng người dùng, dựa trên danh sách bài hát có điểm dự đoán cao nhất.

## **3.9. Quy trình gợi ý Top-N** {#3.9.-quy-trình-gợi-ý-top-n}

Sau khi có mô hình đã train, hệ thống gợi ý cho một user theo quy trình:

1. Lấy hồ sơ user từ bảng `user_profiles` trong SQLite.  
2. Lấy danh sách bài hát từ bảng `songs` trong SQLite.  
3. Loại bỏ các bài user đã tương tác trong bảng `interactions`.  
4. Tạo feature cho từng cặp (user, track).  
5. Dùng model dự đoán predicted\_rating.  
6. Sắp xếp bài hát theo predicted\_rating giảm dần.  
7. Lấy Top-N bài hát làm kết quả gợi ý.

# 

# **CHƯƠNG 4\. KẾT QUẢ THỰC NGHIỆM** {#chương-4.-kết-quả-thực-nghiệm}

## **4.1. Kết quả tiền xử lý dữ liệu** {#4.1.-kết-quả-tiền-xử-lý-dữ-liệu}

Sau khi chạy:

python preprocess\_songs.py \--song-limit 12000

Project tạo được:

12000 bài hát

16 genre chính/top-level

134 genre chi tiết/sub-genre

Một số genre chính có số lượng bài hát nhiều nhất:

| Genre | Số bài |

|---|---:|

| Electronic | 2778 |

| Rock | 1831 |

| Instrumental | 1053 |

| Hip-Hop | 1000 |

| Pop | 1000 |

| International | 1000 |

| Experimental | 1000 |

| Folk | 1000 |

| Classical | 509 |

| Old-Time / Historic | 320 |

## **4.2. Kết quả sinh dữ liệu người dùng và interaction** {#4.2.-kết-quả-sinh-dữ-liệu-người-dùng-và-interaction}

Sau khi chạy:

python generate\_synthetic\_data.py \--users 1200 \--interactions-per-user 35

Dữ liệu được sinh từ bảng `songs` trong SQLite và ghi trực tiếp vào hai bảng `interactions`, `training_pairs`. Bảng `interactions` chỉ lưu `user_id`, `track_id`, `rating`; không lưu cột `liked` vì thông tin thích/không thích có thể suy ra từ rating khi cần.

Ở lần thử nghiệm kiểm tra với 200 user, mỗi user 25 interaction, hệ thống tạo được:

Số interactions: 5000

Số training pairs: 5000

Phân phối rating:

| Rating | Số lượng |

|---:|---:|

| 1 | 207 |

| 2 | 432 |

| 3 | 876 |

| 4 | 1246 |

| 5 | 2239 |

Phân phối mới có đủ rating thấp, trung bình và cao, giúp dữ liệu huấn luyện hợp lý hơn so với việc phần lớn rating chỉ tập trung ở mức 4–5.

## **4.3. Kết quả 2 mô hình** {#4.3.-kết-quả-2-mô-hình}

### ***4.3.1. RandomForestRegressor***  {#4.3.1.-randomforestregressor}

![][image1]

### 

### ***4.3.2. LightGBMRegressor*** {#4.3.2.-lightgbmregressor}

![][image2]

## **4.4. Nhận xét kết quả** {#4.4.-nhận-xét-kết-quả}

### ***4.4.1. RandomForestRegressor***  {#4.4.1.-randomforestregressor}

RMSE \= 0.4672 và MAE \= 0.3888 cho thấy RandomForestRegressor cũng dự đoán rating tốt trên tập dữ liệu hiện tại. RMSE của RandomForestRegressor thấp hơn LightGBMRegressor một chút, nghĩa là mô hình này có lợi thế nhẹ khi xét theo sai số bình phương trung bình.

Precision@10 đạt khoảng 0.9406, Recall@10 đạt khoảng 0.8144 và NDCG@10 đạt khoảng 0.9872. Các chỉ số ranking này rất gần với LightGBMRegressor, cho thấy RandomForestRegressor cũng tạo được danh sách gợi ý Top-N có chất lượng tốt.

### ***4.4.2. LightGBMRegressor*** {#4.4.2.-lightgbmregressor}

RMSE \= 0.4746 và MAE \= 0.3820 cho thấy mô hình LightGBMRegressor dự đoán rating khá sát với rating mô phỏng. MAE thấp hơn RandomForestRegressor một chút, nghĩa là xét theo sai số tuyệt đối trung bình, LightGBM có kết quả tốt hơn nhẹ.

Precision@10 đạt khoảng 0.9417, nghĩa là trong 10 bài hát được gợi ý đầu tiên, trung bình có hơn 94% bài được xem là phù hợp. Recall@10 đạt khoảng 0.8154, cho thấy hệ thống tìm lại được phần lớn các bài hát phù hợp trong tập đánh giá. NDCG@10 đạt khoảng 0.9875, thể hiện các bài hát phù hợp thường được xếp ở vị trí cao trong danh sách gợi ý.

### ***4.4.3. So sánh 2 mô hình*** {#4.4.3.-so-sánh-2-mô-hình}

RandomForestRegressor có RMSE thấp hơn một chút, cho thấy mô hình này giảm được một phần sai số lớn trong dự đoán rating. Ngược lại, LightGBMRegressor có MAE thấp hơn và các chỉ số Precision@10, Recall@10, NDCG@10 cao hơn nhẹ. Điều này cho thấy LightGBMRegressor có lợi thế nhỏ về chất lượng danh sách gợi ý Top-N.

Nhìn chung, hai mô hình cho kết quả khá gần nhau. RandomForestRegressor phù hợp làm mô hình baseline mạnh, dễ giải thích và ổn định. LightGBMRegressor phù hợp làm mô hình chính cho hệ gợi ý vì có hiệu quả ranking nhỉnh hơn và thường có tốc độ inference tốt trên dữ liệu bảng. Trong web demo, hệ thống hiển thị lần lượt kết quả từ cả hai mô hình để người dùng dễ quan sát sự khác biệt.

## **4.5. Triển khai web demo** {#4.5.-triển-khai-web-demo}

Triển khai các chức năng:

- đăng ký và đăng nhập bằng tài khoản thật, mật khẩu được hash trước khi lưu,  
- lưu tài khoản trong bảng `accounts` và hồ sơ cá nhân trong bảng `user_profiles`,  
- nhập/cập nhật thông tin cá nhân gồm tuổi, giới tính, ngôn ngữ nhạc, thể loại yêu thích,  
- cập nhật sở thích âm thanh bằng các thanh kéo có mô tả tiếng Việt dễ hiểu,  
- duyệt danh sách bài hát từ bảng `songs`,  
- xem gợi ý cá nhân hóa từ nút “Xem gợi ý cá nhân” ở trang bài hát,  
- xem chi tiết bài hát,  
- xem hồ sơ user và cập nhật lại thông tin bằng modal,  
- xem dashboard admin.

# **CHƯƠNG 5\. KẾT LUẬN** {#chương-5.-kết-luận}

## **5.1. Kết quả đạt được** {#5.1.-kết-quả-đạt-được}

Đề tài đã xây dựng được một hệ gợi ý nhạc dựa trên thông tin cá nhân bằng cách kết hợp dữ liệu bài hát từ FMA với hồ sơ người dùng và dữ liệu tương tác mô phỏng. Hệ thống sử dụng các đặc trưng về người dùng, bài hát và độ khớp giữa người dùng \- bài hát để huấn luyện mô hình dự đoán rating.

Các kết quả chính đã đạt được:

- Xử lý được dữ liệu FMA metadata và audio features.  
- Tiền xử lý dữ liệu bài hát và lưu trực tiếp vào SQLite `data/musics.db`, bảng `songs`.  
- Sinh được hồ sơ người dùng và dữ liệu interaction/rating mô phỏng.  
- Thiết kế được bộ đặc trưng gồm user features, song features và matching features.  
- Giới thiệu và triển khai thành công hai mô hình RandomForestRegressor và LightGBMRegressor  
- Đánh giá mô hình bằng RMSE, MAE, Precision@10, Recall@10, NDCG@10.  
- Xây dựng web demo Flask để minh họa hệ thống gợi ý.

## **5.2. Hạn chế** {#5.2.-hạn-chế}

\- Dữ liệu user và rating hiện tại là mô phỏng, chưa phải dữ liệu hành vi thực tế.

\- Rating distribution đang thiên về rating cao, có thể làm metrics đẹp hơn thực tế.

\- Audio features như energy, valence, danceability, tempo đang là proxy heuristic từ FMA features, chưa phải giá trị chuẩn như Spotify API.

- Dữ liệu interaction/rating vẫn là dữ liệu mô phỏng, chưa phản ánh hoàn toàn hành vi người dùng thật trong môi trường triển khai thực tế.

\- Chưa có cơ chế cập nhật model từ interaction thực tế sau khi user sử dụng web.
