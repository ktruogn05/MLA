# Kế hoạch thực hiện đồ án Học máy nâng cao

> **Trạng thái mã nguồn 19/10/2026:** đã chuẩn bị pipeline full tại `mla/`, cấu hình `configs/full.json`, tests và notebook demo. Chưa chạy pipeline/tests trong lần bàn giao. Xem [README.md](README.md) để chạy trên máy khác và [IMPLEMENTATION.md](IMPLEMENTATION.md) để xem quyết định kỹ thuật. H₂GCN được triển khai lại bằng PyTorch dựa trên kiến trúc/mã tác giả, không phải chạy nguyên bản TensorFlow. Các ô thực nghiệm bên dưới chỉ được đánh dấu sau khi chạy và kiểm chứng.

> Chuyển từ [main.tex](main.tex) thành checklist triển khai. `main.tex` là bản đề xuất học thuật; file này dùng để theo dõi công việc. Các ô chưa đánh dấu là việc dự kiến, không phải kết quả đã hoàn thành. Khi thay đổi phạm vi hoặc giao thức, cập nhật cả hai file.

## 1. Mục tiêu và câu hỏi nghiên cứu

**Đề tài:** GNN lai cục bộ–không cục bộ với cổng theo nút dựa trên proxy homophily không nhãn, phục vụ phân loại nút trên đồ thị heterophilic.

| Câu hỏi | Bằng chứng cần thu thập |
| --- | --- |
| Proxy tương đồng đặc trưng có phản ánh homophily và lợi ích của nhánh cục bộ không? | Tương quan proxy–homophily, proxy–chênh lệch loss của hai baseline |
| Thêm proxy vào cổng có giúp phân loại tốt hơn không? | So sánh learned gate, proposed gate, shuffled proxy và nối proxy vào MLP |
| Top-k có đánh đổi chất lượng–chi phí hợp lý không? | So sánh Top-k, random-k, local-only; đo retrieval, thời gian và bộ nhớ |

**Tiêu chí thành công:** trả lời các câu hỏi bằng thực nghiệm có đối chứng và tái lập được. Không đặt điều kiện mô hình đề xuất phải thắng mọi baseline. Kết luận chỉ áp dụng cho các dữ liệu và thiết lập đã thử.

## 2. Phạm vi

### Bắt buộc

- [ ] Hai bộ dữ liệu: Roman-empire và Minesweeper.
- [ ] Bảy phương pháp trong bảng ở Mục 4.
- [ ] Ablation bỏ proxy, hoán vị proxy, random-k và nối proxy vào MLP.
- [ ] Phân tích proxy, cổng, lợi ích hai nhánh và chi phí.
- [ ] Notebook demo Minesweeper từ checkpoint đã huấn luyện.
- [ ] Báo cáo, cấu hình, log và hướng dẫn tái lập.

### Mở rộng khi còn tài nguyên

- [ ] Thay GCN bằng GAT để kiểm tra độ nhạy.
- [ ] Full attention trên đồ thị đủ nhỏ.
- [ ] Loại cả lân cận hai bước khỏi tập ứng viên.
- [ ] Ràng buộc gamma không âm bằng softplus.
- [ ] Approximate retrieval, kèm kiểm tra chất lượng so với exact trên mẫu nút.
- [ ] Chạy thêm ba seed cho cấu hình đã khóa.

## 3. Luồng mô hình cần cài đặt

1. Nhận ma trận đặc trưng `X`, cấu trúc đồ thị `A` và mask chia tập.
2. Chuẩn hóa L2 đặc trưng cho retrieval và proxy; vector không được giữ bằng không.
3. Tính proxy `s_v`: trung bình cosine giữa nút v và các láng giềng một bước. Nút cô lập có `s_v = 0`.
4. Chuẩn hóa proxy và `log(1 + degree)` trên toàn đồ thị, epsilon `1e-8`; biến hằng được ánh xạ về 0.
5. Với từng nút, lấy tối đa k ứng viên tương đồng nhất ngoài chính nó và láng giềng trực tiếp. Precompute exact blockwise Top-k, cố định cách phá hòa bằng seed.
6. Encoder tuyến tính kèm ReLU ánh xạ đặc trưng gốc sang embedding chung.
7. Nhánh GCN tạo `z_local`; attention một đầu trên tập ứng viên tạo `z_global`, cùng chiều `d_h`. Tập ứng viên rỗng cho `z_global = 0`.
8. Tính cổng và biểu diễn cuối:

```text
alpha_v = sigmoid(gamma * standardized_proxy_v
                  + gate_mlp([standardized_degree_v, z_local_v, z_global_v]))
z_v = alpha_v * z_local_v + (1 - alpha_v) * z_global_v
```

9. Bộ phân loại tuyến tính tạo logits. Tối ưu CE cho đa lớp hoặc BCE cho nhị phân, cộng phạt L2; loss chỉ tính trên train.

Gate MLP có một lớp ẩn 32 chiều với ReLU và một đầu ra vô hướng. Khởi tạo gamma bằng 0, cho phép học cả dấu âm và dương. Top-k và proxy cố định trong quá trình huấn luyện. Công thức đầy đủ và ký hiệu tham chiếu trong `main.tex`.

## 4. Dữ liệu và mô hình đối chứng

| Dữ liệu | Bài toán | Chỉ số chính | Chỉ số bổ sung |
| --- | --- | --- | --- |
| Roman-empire | Đa lớp | Accuracy | Macro-F1 |
| Minesweeper | Nhị phân, nhãn 0/1 | ROC-AUC | Macro-F1; accuracy và PR-AUC khi cần |

Giữ dữ liệu và mask benchmark. Dùng 10 split theo tỷ lệ train/validation/test 50/25/25 như bản đề xuất. Minesweeper có đặc trưng rời rạc và bị ẩn một phần; giữ nguyên mã hóa của bản phát hành, không khôi phục đặc trưng bằng nhãn.

| Phương pháp | Vai trò |
| --- | --- |
| MLP | Chỉ sử dụng đặc trưng nút |
| GCN local-only | Đối chứng thông tin cục bộ |
| Attention global-only | Đối chứng thông tin không cục bộ |
| Fixed fusion | Hai nhánh, alpha cố định bằng 0.5 |
| Learned gate | Hai nhánh, gamma cố định bằng 0 |
| Proposed gate | Hai nhánh và đường proxy tường minh |
| H₂GCN (PyTorch reimplementation) | Đối chứng chuyên cho heterophily; port độc lập, chưa kiểm tra tương đương mã TensorFlow tác giả |

## 5. Giao thức thực nghiệm

### Quy tắc dữ liệu

- [ ] Chỉ dùng nhãn train cho cập nhật tham số.
- [ ] Chỉ dùng validation cho early stopping và lựa chọn cấu hình.
- [ ] Khóa cấu hình và các lựa chọn thực nghiệm trước khi mở kết quả test.
- [ ] Chỉ tính oracle homophily dùng nhãn ngoài train ở bước hậu nghiệm.
- [ ] Không chọn lại seed, dataset hoặc cấu hình dựa trên kết quả test.
- [ ] Tính homophily trên cạnh vô hướng không trùng, không có self-loop; self-loop của GCN được thêm riêng.

Đặc trưng và cấu trúc toàn đồ thị được phép sử dụng vì đây là thiết lập transductive. Nhãn thật không được đưa vào proxy, retrieval hoặc gate.

### Ngân sách và siêu tham số

| Thành phần | Kế hoạch trong bản đề xuất |
| --- | --- |
| Tìm kiếm | Random search 12 cấu hình/mô hình/bộ dữ liệu |
| Chọn cấu hình | Validation trung bình trên ba split đầu tiên |
| Đánh giá cuối | Một cấu hình chung đã khóa, chạy trên cả 10 split |
| Seed | Một seed huấn luyện cố định/split, công bố trước và dùng chung giữa phương pháp |
| Hidden dimension | 64, 128 |
| Learning rate | 0.001, 0.003 |
| Dropout | 0, 0.3, 0.5 |
| Hệ số L2 | 0, 0.00001, 0.0001 |
| Số lớp GCN | 1, 2, 3 |
| k | 8, 16, 32 |
| Optimizer | Adam; phạt L2 trực tiếp trong loss, không cộng weight decay riêng |
| Epoch tối đa | 1000 |
| Early stopping | Patience 100 theo chỉ số validation chính; hòa điểm chọn checkpoint sớm hơn |
| Nhãn nhị phân để tính F1/accuracy | Ngưỡng 0.5; ROC-AUC tính từ điểm liên tục |

Các siêu tham số chỉ áp dụng cho mô hình có thành phần tương ứng. H₂GCN dùng không gian riêng giới hạn trong `configs/full.json`, tham khảo cài đặt tác giả, cùng số lượt thử. Residual và LayerNorm được đưa vào search của các mô hình có nhánh local; thứ tự cụ thể ghi trong `IMPLEMENTATION.md`.

**Cần chốt trước khi triển khai:** phần cứng, thời gian chạy cho phép, phiên bản dữ liệu/mã nguồn, danh sách seed, vị trí dropout và thứ tự residual/normalization. Đây là các chi tiết chưa được đặc tả hoàn toàn trong đề xuất.

Đo thử ở tuần 1 để xác nhận ngân sách. Nếu không khả thi, thu gọn số cấu hình hoặc split đồng đều giữa các phương pháp trước khi xem test, ghi rõ thay đổi trong cả kế hoạch và báo cáo. Không bỏ lượt có kết quả kém.

### Ablation và phân tích

- [ ] Bỏ proxy: giữ nguyên gate MLP và cố định gamma bằng 0.
- [ ] Shuffled proxy: hoán vị một lần/seed, giữ cố định suốt lượt chạy; không thay retrieval.
- [ ] Random-k: lấy không hoàn lại từ đúng tập ứng viên hợp lệ, cùng số ứng viên và cố định theo seed.
- [ ] Concatenated proxy: nối proxy vào đầu vào MLP thay cho đường cộng riêng; báo cáo số tham số.
- [ ] Phân tích độ nhạy theo k với các giá trị đã định trước.

Bảng mô hình chính dùng tinh chỉnh độc lập với ngân sách bằng nhau. Bảng ablation kiểm tra cơ chế giữ cấu hình proposed gate đã chọn và huấn luyện lại từng biến thể từ đầu; trình bày riêng hai loại so sánh.

- [ ] Báo cáo mean ± độ lệch chuẩn mẫu theo split và số lượt chạy thực tế.
- [ ] So sánh proposed/learned gate theo cặp split: chênh lệch, trung bình chênh lệch, số split cải thiện.
- [ ] Tính `delta_v = loss_global_v - loss_local_v` từ hai baseline được huấn luyện riêng; delta dương nghĩa là baseline local có loss thấp hơn.
- [ ] Tính Spearman proxy–delta và proxy–homophily trên các nút test phù hợp; homophily yêu cầu bậc dương. Biến hằng ghi tương quan không xác định.
- [ ] Vẽ delta trung bình theo nhóm phân vị proxy xác định từ dữ liệu không nhãn.
- [ ] Vẽ phân bố alpha; báo cáo tỷ lệ alpha < 0.05 và alpha > 0.95.
- [ ] Đo retrieval riêng, thời gian/epoch, tổng thời gian, số tham số và bộ nhớ đỉnh nếu đo được.

Không xem các split chồng lấp là các đồ thị độc lập. Alpha là hệ số trộn embedding, không phải xác suất nhánh local đúng. Tương quan proxy–homophily thấp và lợi ích phân loại của proxy là hai vấn đề cần kết luận riêng.

## 6. Lộ trình sáu tuần

### Tuần 1 — Dữ liệu và baseline ban đầu

- [ ] Tải dữ liệu từ nguồn benchmark và ghi phiên bản.
- [ ] Kiểm tra kích thước đặc trưng, nhãn, mask không giao nhau, cạnh trùng, self-loop và nút cô lập.
- [ ] Ghi thống kê cấu trúc; thống kê nhãn tuân theo giai đoạn được phép truy cập.
- [ ] Cài loader, MLP và GCN; chạy thử train/validation.
- [ ] Đo thời gian, bộ nhớ và chốt ngân sách thực tế.

**Đầu ra:** pipeline dữ liệu, baseline chạy được, cấu hình môi trường và ngân sách đã ghi nhận.

### Tuần 2 — Kiến trúc đề xuất

- [ ] Cài chuẩn hóa, proxy, degree feature và cache Top-k.
- [ ] Cài attention, fixed fusion, learned gate và proposed gate.
- [ ] Kiểm tra chiều tensor, mask loss, vector không, nút cô lập và ứng viên rỗng.
- [ ] Kiểm tra các ứng viên không chứa chính nút hoặc láng giềng trực tiếp.
- [ ] Kiểm tra nhãn validation/test không tham gia cập nhật tham số.
- [ ] Lưu và nạp checkpoint cho suy luận.

**Đầu ra:** các biến thể mô hình chạy được và kiểm tra kỹ thuật đạt yêu cầu.

### Tuần 3 — Đối chứng và lựa chọn cấu hình

- [ ] Tích hợp H₂GCN với cùng split và chỉ số.
- [ ] Chốt không gian tìm kiếm, seed và cách ghi log.
- [ ] Chạy random search theo validation cho bảy phương pháp.
- [ ] Chọn cấu hình và lưu lịch sử thử, kể cả lỗi tài nguyên.
- [ ] Chốt danh sách ablation và quy tắc phân tích trước khi xem test.

**Đầu ra:** cấu hình được chọn cho từng mô hình/dataset và giao thức đã khóa.

### Tuần 4 — Đánh giá chính

- [ ] Chạy cấu hình đã khóa trên 10 split hoặc giao thức thu gọn đã công bố.
- [ ] Lưu checkpoint và dự đoán gắn với đúng model, dataset, split, seed.
- [ ] Thu thập kết quả test cuối cùng và chi phí.
- [ ] Tổng hợp bảng mô hình chính; kiểm tra số lượt thiếu/thất bại.

**Đầu ra:** bảng kết quả chính có thể truy ngược về log từng lượt.

### Tuần 5 — Ablation và giải thích kết quả

- [ ] Chạy các ablation bắt buộc theo cấu hình đã chốt.
- [ ] Hoàn thành phân tích tương quan, delta, cổng và độ nhạy theo k.
- [ ] Ghi nhận cả trường hợp có cải thiện và không cải thiện.
- [ ] Viết kết luận cho từng câu hỏi nghiên cứu và giới hạn bằng chứng.

**Đầu ra:** bảng ablation, biểu đồ phân tích và nhận xét dựa trên số liệu thực.

### Tuần 6 — Demo và báo cáo

- [ ] Hoàn thiện demo theo Mục 7.
- [ ] Kiểm tra notebook chạy lại từ đầu bằng checkpoint đã lưu.
- [ ] Chụp giao diện và ghi một lượt thao tác minh họa.
- [ ] Cập nhật báo cáo theo Mục 8.
- [ ] Hoàn thiện hướng dẫn tái lập, slide và video nếu cần.

**Đầu ra:** bộ bàn giao có thể chạy và trình bày được.

## 7. Checklist demo Minesweeper

Demo dùng dữ liệu benchmark và checkpoint có sẵn, không huấn luyện lại trong buổi trình bày.

- [ ] Kiểm tra ánh xạ node ID sang tọa độ bàn cờ.
- [ ] Nạp dataset, split và checkpoint; suy luận trên toàn đồ thị.
- [ ] Hiển thị xác suất có mìn bằng màu, phân biệt train/validation/test.
- [ ] Cho chọn nút test và hiển thị đặc trưng quan sát được.
- [ ] Tô sáng láng giềng một bước và các ứng viên Top-k; ghi rõ GCN nhiều lớp có vùng tiếp nhận rộng hơn.
- [ ] Hiển thị s_v, alpha_v và beta_vu từ lần suy luận thực tế.
- [ ] So sánh dự đoán của local-only, global-only, learned gate và proposed gate trên cùng nút/split.
- [ ] Có chế độ bật nhãn thật sau dự đoán; minh họa cả đúng và sai.
- [ ] Hiển thị bảng ROC-AUC, ablation và biểu đồ đã thu được từ thực nghiệm.
- [ ] Ghi rõ kết quả từng nút minh họa khác với kết quả tổng hợp toàn tập test.

Giữ nguyên đặc trưng benchmark; không biến thao tác bấm ô thành cập nhật đặc trưng như trò chơi dò mìn. Không dùng nhãn thật cho suy luận. Không diễn giải attention như bằng chứng nhân quả.

## 8. Đưa kết quả thực tế vào báo cáo

| Nội dung có sau triển khai | Vị trí trong báo cáo cuối kỳ |
| --- | --- |
| Bảng baseline, ablation, chi phí, biểu đồ phân tích | Kết quả thực nghiệm và thảo luận; cập nhật phần kết quả dự kiến thành kết quả đã đo |
| Ảnh giao diện, lượt thao tác, dự đoán và cổng thực tế | Đổi “Kịch bản demo dự kiến” thành “Demo hệ thống và minh họa kết quả”, đặt sau kết quả thực nghiệm |
| Notebook, mã nguồn, checkpoint, cách chạy, video | Phụ lục hướng dẫn tái lập, dẫn tới sản phẩm tồn tại và đã kiểm tra |
| Kết luận cho ba câu hỏi nghiên cứu | Phần kết luận, nêu rõ giới hạn và kết quả âm nếu có |

Chỉ ghi số liệu đã đo; không chuyển kỳ vọng thành kết quả. Khi chuyển từ proposal sang báo cáo cuối kỳ, cập nhật cả phần mô tả phương pháp để phản ánh đúng cài đặt thực tế.

## 9. Tổ chức sản phẩm đề xuất

Sơ đồ dưới đây là định hướng ban đầu; cấu trúc mã thực tế dùng package `mla/` và output `runs/`, được mô tả trong README. Dataset/checkpoint/log được tạo khi chạy trên máy đích:

```text
MLA/
  main.tex                 # Đề xuất, sau này cập nhật thành báo cáo
  PLAN.md                  # Checklist theo dõi
  README.md                # Hướng dẫn thiết lập và chạy
  configs/                 # Cấu hình và seed đã chốt
  src/                     # Loader, mô hình, train, evaluate, phân tích
  notebooks/demo.ipynb     # Demo tương tác
  data/                    # Dữ liệu và cache, kèm nguồn/phiên bản
  checkpoints/             # Trọng số mô hình và cấu hình tương ứng
  results/                 # Log, chỉ số, dự đoán từng lượt
  figures/                 # Biểu đồ và ảnh demo dùng cho báo cáo
```

Mỗi lượt chạy cần ghi ít nhất: dataset, phiên bản, model, split, seed, cấu hình, số tham số, checkpoint được chọn, chỉ số validation/test theo đúng giai đoạn, thời gian và lỗi nếu có. Mỗi checkpoint phải truy được cấu hình và tiền xử lý tương ứng.

## 10. Điều kiện hoàn thành

- [ ] Đủ phạm vi bắt buộc hoặc có ghi rõ giao thức thu gọn đã chốt trước test.
- [ ] Kết quả và biểu đồ truy được về log thực nghiệm.
- [ ] Có câu trả lời có bằng chứng cho cả ba câu hỏi nghiên cứu.
- [ ] Demo chạy từ checkpoint và hiển thị đầu ra thật của mô hình.
- [ ] Hướng dẫn tái lập đã được kiểm tra.
- [ ] Báo cáo phân biệt rõ dự kiến, kết quả thực tế và giới hạn.
- [ ] Đồng bộ thay đổi giữa `PLAN.md`, cấu hình thực nghiệm và `main.tex`.

## 11. Tài liệu gốc trong đề xuất

- [GCN](https://arxiv.org/abs/1609.02907)
- [GAT](https://arxiv.org/abs/1710.10903)
- [Geom-GCN](https://arxiv.org/abs/2002.05287)
- [H₂GCN](https://arxiv.org/abs/2006.11468)
- [Benchmark heterophily](https://arxiv.org/abs/2302.11640)
- [Mã nguồn và dữ liệu benchmark](https://github.com/yandex-research/heterophilous-graphs)
- [GraphGPS](https://arxiv.org/abs/2205.12454)
