# Kế hoạch thu gọn cho CPU và RAM 8 GB

Kế hoạch này thay ngân sách lớn trước đây. [PLAN_FULL.md](PLAN_FULL.md) giữ bản cũ để tham khảo, không phải phạm vi bắt buộc hiện tại. Đồng bộ đề xuất học thuật tại [main.tex](main.tex).

**Cập nhật mã nguồn 19/10/2026:** đã viết pipeline dùng chung full/reduced theo yêu cầu, nhưng chưa chạy. Để chạy trên máy khác dùng [README.md](README.md), `configs/full.json` hoặc `configs/reduced.json`. Các checklist dưới đây vẫn là tiêu chí cần chạy và xác nhận; việc đã viết mã không đồng nghĩa thí nghiệm hoàn thành. Preset reduced mới có MLP nhiều lớp và activation checkpointing nên không áp nguyên số đo của pilot cũ.

## 1. Mục tiêu và giới hạn

Kiểm tra proxy tương đồng đặc trưng có giúp cổng kết hợp GCN và attention không cục bộ hay không; phân tích proxy–homophily, lợi ích hai nhánh và chi phí. Kết quả âm vẫn hợp lệ.

Chỉ dùng một dữ liệu, ba split và một cấu hình; chưa có đối chứng chuyên heterophily. Kết luận chỉ áp dụng cho các biến thể trong thiết lập này, không khẳng định vượt benchmark hoặc tổng quát cho mọi đồ thị.

## 2. Tiến độ đã xác nhận

- [x] Môi trường có Python 3.11.6, PyTorch 2.9.1 CPU và các thư viện cần cho pilot.
- [x] Tải Minesweeper chính thức: 10.000 nút, 7 chiều đặc trưng, 39.402 cạnh vô hướng.
- [x] Viết [benchmark_cpu.py](benchmark_cpu.py), dùng phép tính thưa và block để giảm RAM.
- [x] Chạy 30 epoch/model cho MLP, GCN, fixed fusion, learned gate và proposed gate trên split 0, seed 42.
- [x] Lưu số đo, môi trường và SHA-256 dữ liệu tại [results/cpu_pilot.json](results/cpu_pilot.json).
- [ ] Hoàn thiện pipeline chính với early stopping, checkpoint và đánh giá test.
- [ ] Chạy thực nghiệm thu gọn, phân tích, demo và báo cáo.

Pilot chỉ đo train + validation; chưa lưu checkpoint mô hình và chưa đánh giá test. Số đo chi tiết nằm trong [CPU_BENCHMARK.md](CPU_BENCHMARK.md).

## 3. Cấu hình thực nghiệm cố định

| Thành phần | Thiết lập |
| --- | --- |
| Dữ liệu | Minesweeper đầy đủ, giữ nguyên đặc trưng benchmark |
| Split | Mask chính thức 0, 1, 2; tỷ lệ train/validation/test 50/25/25 |
| Seed huấn luyện | 42, 43, 44 tương ứng từng split; chung giữa các mô hình |
| Tìm kiếm siêu tham số | Không chạy; một cấu hình |
| CPU | 2 luồng tính toán, interop 1 |
| Hidden dimension | 64 |
| Encoder | Linear + ReLU |
| GCN | 2 lớp, ReLU, dropout 0.3 sau mỗi lớp |
| Attention | Một đầu, chiều 64, k = 8, không dropout |
| Gate | Lớp ẩn 32 chiều, ReLU; gamma khởi tạo 0, không ràng buộc dấu |
| Residual / LayerNorm | Không dùng |
| Optimizer | Adam, learning rate 0.001, lambda = 0, không weight decay |
| Giới hạn mỗi lượt | 200 epoch, patience 30; hòa validation chọn checkpoint sớm hơn |
| Chỉ số chính | ROC-AUC; validation chọn checkpoint, test đánh giá cuối |
| Bổ sung | Macro-F1 tại ngưỡng 0.5, số tham số, thời gian và bộ nhớ |
| Retrieval | Exact cosine Top-k, block 64 nút; phá hòa bằng thứ tự nút xáo trộn seed 42 |
| Attention theo block | 256 nút/block |

Chuẩn hóa L2 chỉ cho proxy/retrieval. Proxy là cosine trung bình trên lân cận; nút cô lập nhận 0. Chuẩn hóa proxy và log(1 + degree) trên toàn đồ thị, epsilon 1e-8. Loại chính nút và láng giềng một bước khỏi tập ứng viên; ứng viên rỗng cho embedding global bằng 0. Nhãn không tham gia tạo proxy hoặc retrieval.

## 4. Ma trận chạy: 27 lượt

| Biến thể | Số lượt | Mục đích |
| --- | --- | --- |
| MLP | 3 | Chỉ đặc trưng |
| GCN local-only | 3 | Thông tin cục bộ |
| Attention global-only | 3 | Thông tin không cục bộ |
| Fixed fusion | 3 | Alpha = 0.5 |
| Learned gate | 3 | Gamma = 0, đối chứng bỏ proxy |
| Proposed gate | 3 | Proxy tường minh |
| Shuffled proxy | 3 | Giá trị của việc gắn proxy đúng nút |
| Random-k | 3 | Giá trị của retrieval theo similarity |
| Proxy nối vào MLP | 3 | Cách đưa proxy vào cổng |
| **Tổng** | **27** | Tối đa 5.400 epoch |

Mỗi biến thể được huấn luyện lại từ đầu. Bỏ proxy đã có learned gate, không chạy trùng. Shuffled proxy cố định một lần/seed, giữ retrieval nguyên vẹn. Random-k lấy không hoàn lại từ đúng tập ứng viên hợp lệ và cố định theo seed. Báo cáo số tham số của biến thể nối proxy.

## 5. Checklist triển khai tiếp

### A. Hoàn thiện mã huấn luyện

- [ ] Tách loader, mô hình, train/evaluate từ pilot thành mã tái sử dụng.
- [ ] Thêm global-only và ba ablation còn thiếu; chốt cấu trúc MLP/global-only.
- [ ] Kiểm tra mask, tensor, gradient, vector không, nút cô lập và ứng viên rỗng.
- [ ] Thêm checkpoint validation tốt nhất, early stopping và log từng lượt.
- [ ] Lưu dataset hash, phiên bản, split, seed, cấu hình và lỗi tài nguyên.
- [ ] Khóa lựa chọn và danh sách phân tích trước khi xem test.

### B. Thực nghiệm chính

- [ ] Chạy tuần tự từng model/split, tái sử dụng proxy và retrieval cache.
- [ ] Chỉ cập nhật tham số từ nhãn train; chỉ chọn checkpoint bằng validation.
- [ ] Đánh giá test sau khi khóa các lựa chọn, không đổi seed/cấu hình vì test kém.
- [ ] Báo cáo mean ± độ lệch chuẩn mẫu và từng kết quả của ba split.
- [ ] So sánh proposed/learned theo cặp split; không coi các split chồng lấp là đồ thị độc lập.
- [ ] Ghi riêng thời gian retrieval, epoch, tổng thời gian, RAM và số tham số.

### C. Phân tích hậu nghiệm

- [ ] Tính oracle homophily sau khi khóa thí nghiệm; node homophily chỉ trên nút bậc dương.
- [ ] Tính delta = loss_global − loss_local từ hai baseline huấn luyện riêng.
- [ ] Tính Spearman proxy–homophily và proxy–delta; biến hằng ghi không xác định.
- [ ] Vẽ delta theo nhóm phân vị proxy, phân bố alpha và tỷ lệ alpha < 0.05 hoặc > 0.95.
- [ ] Kết luận riêng về khả năng phản ánh homophily và lợi ích phân loại của proxy.

### D. Demo và báo cáo

- [ ] Notebook nạp checkpoint, suy luận toàn đồ thị, hiển thị vùng bàn cờ Minesweeper.
- [ ] Kiểm tra ánh xạ node ID–tọa độ trước khi vẽ.
- [ ] Chọn nút test để xem đặc trưng, lân cận, Top-k, proxy, alpha và attention.
- [ ] So sánh local-only, global-only, learned và proposed trên cùng nút/split.
- [ ] Bật nhãn thật riêng sau dự đoán; minh họa cả đúng và sai.
- [ ] Hiển thị bảng và biểu đồ từ số liệu thực nghiệm thực tế.
- [ ] Đưa bảng đánh giá vào “Kết quả thực nghiệm và thảo luận”; đổi mục demo dự kiến thành demo thực tế phía sau.
- [ ] Đưa mã nguồn, checkpoint và hướng dẫn chạy vào phụ lục tái lập.

Alpha là hệ số trộn embedding, không phải xác suất local đúng. Demo giữ nguyên đặc trưng benchmark, không cập nhật theo thao tác mở ô. Không dùng nhãn test làm đầu vào.

## 6. Phần mở rộng chưa chạy

Roman-empire, H₂GCN, random search, tăng lên 10 split, độ nhạy k, GAT, full attention, loại lân cận hai bước, gamma không âm và approximate retrieval. Chỉ bổ sung sau phạm vi chính và ghi rõ giao thức; nếu thiết kế lại dựa trên test đã xem, phải thừa nhận giới hạn đánh giá.

## 7. Chạy lại phép đo

```powershell
python benchmark_cpu.py --epochs 30
```

Lệnh dùng `data/minesweeper.npz` đã tải và ghi lại `results/cpu_pilot.json`. Thời gian phụ thuộc ứng dụng đang mở, nguồn điện và nhiệt độ máy. Các ô checklist chưa đánh dấu chưa được hoàn thành.
