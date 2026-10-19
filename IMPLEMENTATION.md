# Trạng thái triển khai và các quyết định kỹ thuật

Cập nhật 19/10/2026 theo yêu cầu viết mã full plan nhưng chưa chạy. Mã Python, JSON config và notebook đã được chuẩn bị; kiểm tra tĩnh không đồng nghĩa kiểm chứng runtime hoặc kết quả thực nghiệm. Tests được viết để chạy trên máy đích.

## Ánh xạ plan → mã nguồn

| Yêu cầu | Cài đặt |
| --- | --- |
| Hai dataset chính thức | Loader `Graph`, downloader pin commit nguồn khi tải mới |
| Không nhãn trong proxy/retrieval | `features`, `candidates` chỉ dùng X và adjacency |
| Exact Top-k ngoài một bước | Blockwise cosine, stable sort theo permutation seed cố định, padding -1 |
| Biến thể không có ứng viên | Attention trả vector 0; masked softmax không NaN |
| Bảy baseline | `Classifier`, tên `mlp/gcn/global/fixed/learned/proposed/h2gcn` |
| Bốn ablation | `variants`: no_proxy, shuffled_proxy, random_k, concat_proxy |
| Đối chứng fixed-config | proposed_control riêng trong bảng ablation; checkpoint trùng được tái sử dụng |
| Random search 12 cấu hình | `configurations`, unique, chỉ lấy trung bình validation trên ba split |
| Early stopping và resume | `best.pt`, `last.pt`, optimizer/RNG/history/patience |
| Khóa trước test | `LOCK.json`, complete fit manifest và checkpoint hashes |
| So sánh từng split | CSV, sample std, paired differences; không kiểm định coi nút độc lập |
| Phân tích hậu nghiệm | `analysis.py`, chỉ sau COMPLETE evaluation |
| Demo thật | `demo.py` + notebook, checkpoint và predictions thật, không dữ liệu giả |
| Mở rộng | GAT, strict-twohop exclusion, positive gamma, full attention, ANN, extra seeds |

## Chi tiết đã chốt khi viết mã

- MLP có encoder Linear–ReLU và `layers` lớp hidden Linear–ReLU–Dropout, sau đó output tuyến tính.
- GCN có encoder riêng, `layers` bước normalized sparse adjacency với self-loop và projection không bias. Mỗi bước: aggregation → residual nếu bật → LayerNorm nếu bật → ReLU → dropout.
- GAT là một đầu với LeakyReLU 0.2, softmax theo nút nhận, có self-loop. Đây là mở rộng về backbone, không phải baseline GAT tune độc lập.
- Global-only dùng encoder và attention, không nhánh GCN; các siêu tham số layers/dropout local không được tìm kiếm cho mô hình này.
- Attention một đầu có Q/K/V không bias và không attention dropout. Activation checkpoint theo block tái tính khi backward để giảm activations giữ trong bộ nhớ.
- Gate hidden 32, ReLU, một logit. Learned gate không có tham số gamma học được; proposed thêm một scalar. Concatenated proxy tăng chiều đầu vào đúng một.
- Với positive gamma dùng softplus(-5) lúc khởi tạo, do softplus không thể bằng 0 hữu hạn.
- Nhị phân dùng một logit và BCE; đa lớp dùng CE. L2 phạt mọi tham số trainable theo công thức đề xuất, Adam không cộng weight decay.
- Chỉ số PR bổ sung là sklearn average precision, ghi đúng tên `average_precision`; không đồng nhất với diện tích trapezoidal PR.
- Retrieval cache dùng dataset hash, phiên bản thuật toán, mode, k, seed và số hop loại trừ. Seed retrieval cố định, seed shuffle proxy theo lượt train.
- Full attention tiêu chuẩn nhìn tất cả nút, gồm chính nó và láng giềng. Vì vậy đối chứng full khác restricted cả về phạm vi và số lượng; cần nêu khi diễn giải.
- Tiền xử lý CPU; train chọn CPU/CUDA. Không có DDP, mixed precision hoặc multi-writer output trong bản này.

## H₂GCN: thay đổi so với ý định chạy nguyên mã tác giả

[Mã gốc](https://github.com/GemsLab/H2GCN/tree/master/h2gcn/models) sử dụng TensorFlow và framework thí nghiệm riêng. Bản hiện tại là **port độc lập sang PyTorch**, không sao chép/vendoring mã gốc. Cấu trúc giữ các điểm chính: projection đầu vào không bias + ReLU; normalized adjacency strict 1-hop/2-hop không ego; nối hai kết quả ở mỗi bước, không thêm activation giữa các bước; nối ego và mọi mức trung gian rồi dropout và classifier không bias.

Không gian riêng trong full config dùng hidden 64/128, depth 1/2, dropout 0/0.5, lr 0.001/0.01 và L2 0/0.0005. Đây là không gian giới hạn do dự án chọn, có giá trị theo gợi ý mã gốc, không phải toàn bộ grid của tác giả.

Giữ feature thô theo benchmark chung thay vì row-normalize mặc định của runner tác giả. Dùng BCE một logit cho Minesweeper, Adam/early stopping chung theo ROC-AUC, thay cho runner nguyên bản. Initializer và thứ tự nối biểu diễn theo PyTorch không được tuyên bố tương đương bit-for-bit TensorFlow. Cần chạy kiểm tra đối chiếu nếu muốn kết luận đã tái hiện H₂GCN chính thức. Trong báo cáo ghi **H₂GCN (PyTorch reimplementation)**.

## Việc chưa được thực hiện

- Chưa tải Roman-empire, cài dependencies mới, chạy tests, forward/backward hoặc huấn luyện pipeline mới.
- Chưa tạo kết quả test, checkpoint final, hình demo hoặc kết luận khoa học.
- Chưa kiểm chứng CUDA, notebook widgets hoặc ANN trên máy đích.
- Việc đưa mã và notebook lên GitHub không thực thi notebook, huấn luyện hoặc triển khai dịch vụ.

Hướng dẫn trong README phải được chạy và xác nhận trên máy đích. Không đánh dấu các mục thực nghiệm của plan là hoàn thành chỉ vì đã có mã nguồn.
