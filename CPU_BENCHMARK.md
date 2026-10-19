# Đo thời gian thực nghiệm trên CPU

Ngày đo: 18/10/2026. Đây là pilot kiểm tra khả năng chạy, chưa phải đánh giá chất lượng cuối cùng.

## Thiết lập thực tế

- Máy người dùng: Intel Core i5-10210U, RAM 8 GB, Intel UHD Graphics. Script sử dụng CPU, không sử dụng GPU.
- Python 3.11.6, PyTorch 2.9.1+cpu, hai luồng tính toán.
- Minesweeper chính thức: 10.000 nút, 7 chiều đặc trưng, 39.402 cạnh vô hướng; split 0, seed 42.
- Hidden dimension 64, GCN hai lớp, k = 8, gate 32 chiều; Adam 0.001; không residual/LayerNorm.
- Tính Top-k exact theo block 64 nút; attention theo block 256 nút.
- Chạy tuần tự năm mô hình, mỗi mô hình 30 epoch. Mỗi epoch gồm train forward, loss, backward, cập nhật Adam, rồi eval forward và tính ROC-AUC validation.
- Chỉ truy xuất nhãn train/validation cho tính toán. Không đánh giá test, không early stopping trong pilot và không lưu checkpoint huấn luyện.

## Kết quả đã đo

| Mô hình | Tổng 30 epoch | Trung vị mỗi epoch sau 3 epoch đầu | Ngoại suy 200 epoch từ tổng 30 epoch |
| --- | ---: | ---: | ---: |
| MLP | 0,41 giây | 0,011 giây | 2,7 giây |
| GCN local-only | 4,82 giây | 0,159 giây | 32,1 giây |
| Fixed fusion | 16,01 giây | 0,498 giây | 106,7 giây |
| Learned gate | 16,98 giây | 0,549 giây | 113,2 giây |
| Proposed gate | 18,18 giây | 0,562 giây | 121,2 giây |

- Tiền xử lý cấu trúc và proxy: **0,23 giây**.
- Exact Top-k: **7,19 giây**, tái sử dụng giữa các mô hình cùng cấu hình.
- Tổng các vòng train/validation: **56,40 giây**; cộng tiền xử lý và retrieval khoảng **63,82 giây**. Không gồm import thư viện, tải dữ liệu từ mạng và khởi tạo ngoài vùng đo.
- Peak RSS của tiến trình quan sát bằng lấy mẫu mỗi 50 ms: **456,9 MiB**. Đây là bộ nhớ tiến trình, không phải toàn bộ RAM máy; lấy mẫu có thể bỏ lỡ đỉnh rất ngắn.
- RAM khả dụng sau lượt đo khoảng **770 MiB**; các ứng dụng khác đang chiếm phần lớn RAM hệ thống.

## Ước lượng ngân sách thu gọn

Kế hoạch mới có 9 biến thể × 3 split = 27 lượt, mỗi lượt tối đa 200 epoch. Nếu tạm tính mọi lượt đều mất bằng proposed gate đã đo, phép ngoại suy là:

```text
27 × 200 × (18,1828 / 30) ≈ 3.273 giây ≈ 54,5 phút
```

Đây là phép tính dự trù, không phải thời gian đã đo cho cả 27 lượt, cũng không phải cận trên được bảo đảm. MLP/GCN nhanh hơn; early stopping có thể giảm thời gian. Global-only và các ablation chưa đo, còn ghi checkpoint, I/O, RAM thiếu và nhiệt độ máy có thể làm tăng thời gian. Nên dành **khoảng 1–2 giờ cho một đợt chạy tính toán thu gọn** rồi cập nhật ước lượng bằng log thật; chưa tính thời gian viết mã, sửa lỗi, phân tích và làm demo.

## Giới hạn và cách diễn giải

Pilot cho thấy cấu hình nhỏ trên toàn bộ Minesweeper chạy được trên CPU. Điều này điều chỉnh nhận định ban đầu: laptop có thể chạy thực nghiệm thu gọn, chưa cần GPU cho phạm vi này. Không suy rộng số đo sang Roman-empire, hidden dimension lớn hoặc tìm kiếm hàng trăm cấu hình.

Validation AUC cuối 30 epoch được lưu trong JSON để kiểm tra kỹ thuật. Mô hình lai chưa cho thấy ưu thế ở lượt ngắn này; không dùng kết quả đó làm kết luận khoa học hoặc chọn lại cấu hình. Chưa có kết quả test.

## Tái lập

```powershell
python benchmark_cpu.py --epochs 30
```

- [Script đo](benchmark_cpu.py)
- [Log JSON đầy đủ](results/cpu_pilot.json)
- [Kế hoạch đang áp dụng](PLAN.md)
- [Dữ liệu nguồn chính thức](https://github.com/yandex-research/heterophilous-graphs/blob/main/data/minesweeper.npz)

SHA-256 của file dữ liệu:

```text
e664c8dacf1e8ac466c2c09ed4b237bd2c5541f47a6eae9c6092cb87f16412b3
```
