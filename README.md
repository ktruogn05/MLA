# GNN local–global: full experiment pipeline

Mã nguồn cho đề tài trong `main.tex`, có preset full và CPU thu gọn. **Bản triển khai mới chưa chạy huấn luyện, chưa chạy test tự động hoặc kiểm chứng trên CUDA.** Kết quả pilot CPU cũ trong `CPU_BENCHMARK.md` thuộc script `benchmark_cpu.py`, không chứng nhận tốc độ/chất lượng pipeline mới.

## 1. Có gì trong repository?

| Thành phần | Vị trí |
| --- | --- |
| Dataset, kiểm tra mask, proxy, sparse graph, Top-k cache | `mla/data.py` |
| MLP, GCN, attention-only, fixed/learned/proposed gate, H₂GCN port | `mla/models.py` |
| Train/validation, early stopping, checkpoint và resume | `mla/train.py` |
| Random search → khóa → fit → test → phân tích | `mla/cli.py` |
| So sánh paired split, oracle homophily, loss delta và biểu đồ | `mla/analysis.py` |
| Notebook tương tác từ checkpoint thật | `notebooks/demo.ipynb` |
| Cấu hình full / thu gọn | `configs/full.json`, `configs/reduced.json` |
| Kiểm tra nhỏ bằng dữ liệu giả | `tests/test_core.py` |

**H₂GCN:** bản PyTorch triển khai lại kiến trúc từ bài báo và mã TensorFlow tác giả, có exact 1-hop/2-hop, tách ego và nối các mức biểu diễn. Không phải chạy nguyên mã tác giả, chưa kiểm tra numerical parity. Không được ghi trong báo cáo rằng đã tái hiện nguyên bản. Xem [IMPLEMENTATION.md](IMPLEMENTATION.md) về các quyết định và khác biệt.

## 2. Cài trên máy khác

Python 3.11 được khuyến nghị. Chạy lệnh từ thư mục repository. Tạo môi trường riêng:

```powershell
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Cài PyTorch **phù hợp phần cứng** trước. Máy CPU:

```powershell
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
```

Máy NVIDIA: lấy lệnh CUDA phù hợp driver từ [PyTorch Get Started](https://pytorch.org/get-started/locally/). Không dùng wheel CPU nếu muốn tăng tốc CUDA. Sau đó:

```powershell
python -m pip install -r requirements.txt
python -m pip freeze > environment-installed.txt
```

Các requirements là khoảng phiên bản, chưa phải lockfile đã kiểm chứng. Giữ `environment-installed.txt` cùng kết quả thực nghiệm. Không cần PyTorch Geometric hoặc TensorFlow cho pipeline mặc định.

## 3. Kiểm tra trước khi chạy dài

Các lệnh dưới đây được chuẩn bị để bạn chạy trên máy đích; chưa được thực thi trong lần bàn giao mã:

```powershell
python -m pytest -q
python -m mla.cli plan --config configs/full.json
```

`plan` chỉ kiểm tra không gian cấu hình và in ngân sách, không tải dữ liệu hoặc huấn luyện. `pytest` dùng đồ thị giả rất nhỏ, có forward/backward để kiểm tra kỹ thuật; không chạy benchmark.

## 4. Chạy plan full theo từng giai đoạn

Sau khi cài đặt và kiểm tra, có thể dùng runner nối các giai đoạn tuần tự:

```powershell
python scripts/run_full.py
```

Lệnh trên chỉ in kế hoạch. Để thực sự tải dữ liệu và chạy toàn bộ trên máy đích:

```powershell
python scripts/run_full.py --run --device auto
```

Runner dừng nếu một giai đoạn lỗi; chạy lại cùng lệnh để tiếp tục. Nếu đã có `LOCK.json`, bỏ qua tune/lock và kiểm tra/tiếp tục các bước sau. Không tự chạy tests. Các lệnh từng bước tương đương ở dưới giúp kiểm tra tiến độ và kết quả trung gian.

### Tải dữ liệu

```powershell
python -m mla.cli download --config configs/full.json
```

Tải từ repo benchmark chính thức, ghi commit nguồn và SHA-256 cho file mới. File đã tồn tại được giữ nguyên và hash được ghi trong manifest thực nghiệm. Không đưa dataset lên GitHub.

### Tìm cấu hình bằng validation

```powershell
python -m mla.cli tune --config configs/full.json --output runs/full --device auto
```

Hai dữ liệu × bảy mô hình × 12 cấu hình × ba split = 504 lượt tìm kiếm. Không tính test. Lỗi/OOM được ghi `failure.json` rồi dừng, không âm thầm loại cấu hình. Chạy lại đúng lệnh để tiếp tục từ checkpoint cuối mỗi epoch. Có thể dùng `--device cuda:0` hoặc `--device cpu --threads 2`.

### Khóa giao thức

```powershell
python -m mla.cli lock --config configs/full.json --output runs/full
```

`LOCK.json` ghi cấu hình thắng theo validation, danh sách ablation, dữ liệu, mã nguồn và hash. Lệnh không huấn luyện và không đọc kết quả test. Mọi lựa chọn extension phải nằm trong config trước khi khóa.

### Huấn luyện final và ablation

```powershell
python -m mla.cli fit --config configs/full.json --output runs/full --device auto
```

Chạy 10 split cho các mô hình chính, ablation theo cấu hình proposed cố định và độ nhạy k = 8/16/32. Các lượt hoàn toàn trùng config/split/seed được tái sử dụng, kể cả lượt tốt nhất đã chạy khi tuning. Vì vậy số lượt thực tế không đơn giản bằng 644 cộng tất cả ablation. Manifest phân biệt bảng chính tinh chỉnh độc lập với ablation cố định cấu hình.

### Đánh giá test một khi tất cả lượt final hoàn thành

```powershell
python -m mla.cli evaluate --config configs/full.json --output runs/full --device auto
python -m mla.cli analyze --config configs/full.json --output runs/full
```

`evaluate` đòi hỏi manifest final đầy đủ và checkpoint chưa thay đổi. `analyze` mới tính oracle homophily từ nhãn toàn đồ thị. Không dùng các đầu ra này để chọn lại cấu hình rồi tuyên bố test độc lập.

## 5. Preset thu gọn

Dùng cùng chuỗi lệnh nhưng thay `--config configs/reduced.json --output runs/reduced`. Preset một dataset, ba split, cấu hình cố định. Các alias ablation trùng cấu hình được tái sử dụng, còn 27 lượt huấn luyện duy nhất nếu không thay config. Pilot cũ không có checkpoint để tái sử dụng.

## 6. Các mở rộng đã có mã, mặc định tắt

Tạo bản sao config và sửa trường `extensions` trước khi tạo output mới:

```json
"extensions": ["gat", "twohop_exclusion", "positive_gamma", "ann"]
```

- `gat`: GAT một đầu cho nhánh local, giữ cấu hình proposed đã chọn; đây là thí nghiệm độ nhạy, không phải GAT đã tune riêng.
- `twohop_exclusion`: loại thêm lân cận strict 2-hop khỏi retrieval.
- `positive_gamma`: gamma = softplus(rho), rho khởi tạo -5 (gần 0 nhưng không bằng 0).
- `ann`: cài `requirements-optional.txt`; ghi recall@k trên mẫu 100 nút so với exact. Hòa điểm khiến recall theo ID nhạy với tie-breaking.
- `full_attention`: self-attention trên mọi nút, gồm self và láng giềng. Có giới hạn `full_max_nodes=12000`; Roman-empire vượt giới hạn sẽ dừng rõ ràng. Để chạy mở rộng này, tạo config chỉ Minesweeper hoặc tăng giới hạn sau khi kiểm tra bộ nhớ; không tự động bỏ qua dataset.
- `extra_seed_bases`: ví dụ `[1042, 2042, 3042]` để chạy thêm ba seed cho main models; bảng tổng hợp lấy trung bình seed trong mỗi split trước khi tính mean/std giữa split.

Full attention vẫn tốn O(N²) tính toán; chunk và activation checkpoint giảm bộ nhớ trung gian chứ không làm phép tính tuyến tính. H₂GCN/two-hop có giới hạn số cạnh để tránh mở rộng đồ thị không kiểm soát; tăng giới hạn phải thực hiện trước khi tạo output mới.

## 7. Resume và chuyển máy

- Chạy lại cùng lệnh/config/output để tiếp tục. `last.pt` chứa model, optimizer, RNG, lịch sử và trạng thái patience; `best.pt` là checkpoint validation tốt nhất.
- Không sửa mã/config/dataset trong output đang chạy: pipeline so hash và yêu cầu output mới nếu khác.
- Chỉ chạy một tiến trình ghi vào một output/cache tại một thời điểm; chưa có hỗ trợ distributed/multi-writer.
- Chuyển máy: sao chép repo cùng phiên bản, dataset, cache và thư mục `runs/full`. CUDA khác phần cứng/backend có thể không tái lập bit-for-bit, dù seed và trạng thái được giữ.
- Checkpoint có Python state; chỉ nạp checkpoint tin cậy do bạn tạo.

## 8. Demo và đầu ra

Sau `evaluate` và `analyze`:

```powershell
jupyter lab notebooks/demo.ipynb
```

Notebook chạy CPU, nạp checkpoint thật, kiểm tra ánh xạ bàn cờ từ cạnh. Chọn nút test để xem proxy, cổng, Top-k, attention và dự đoán của bốn mô hình. Giao diện có nút bật nhãn thật riêng. Nếu ánh xạ row-major không khớp, notebook báo lỗi thay vì vẽ sai.

```text
runs/full/
  experiment.json          # Plan, mã nguồn, dataset hashes
  selection.json           # Cấu hình thắng validation và toàn bộ search
  LOCK.json                # Giao thức khóa trước test
  jobs/<id>/
    metadata.json
    last.pt                # Resume
    best.pt                # Checkpoint validation
    result.json            # Epoch history, validation, timing, memory
    failure.json           # Nếu có lỗi
  manifest.json            # Các vai trò main/ablation/sensitivity và checkpoint hashes
  evaluation/              # Test metrics, predictions; COMPLETE.json
  analysis/                # runs.csv, summary.csv, paired differences, oracle stats, PNG
```

`compute_seconds` đo train + validation; checkpoint I/O nằm ngoài timer epoch. `walltime.json` chỉ đo lần gọi hiện tại, không cộng các phiên đã bị gián đoạn. Peak RSS lấy mẫu là bộ nhớ tiến trình; CUDA peak ghi riêng. Macro-F1, ROC-AUC và **average precision** được ghi đúng tên; average precision không được gọi nhầm là trapezoidal PR-AUC.

## 9. Đưa lên GitHub

`.gitignore` loại data/cache/runs/checkpoints và output lớn. Upload mã nguồn, config, notebook chưa có outputs, tài liệu và tests. Các file dataset/checkpoint sẽ tải hoặc sinh trên máy đích. Giữ riêng `CPU_BENCHMARK.md` nếu muốn lưu lịch sử pilot; đường dẫn JSON pilot sẽ không tồn tại ở clone mới vì results bị ignore.

Repository đích: [ktruogn05/MLA](https://github.com/ktruogn05/MLA). GitHub lưu mã; các lệnh huấn luyện chạy trên máy bạn chọn, không tự chạy trên GitHub. Repo này không có workflow tự động khởi chạy thực nghiệm.

## 10. Nguồn tham khảo

- [Benchmark chính thức và dữ liệu](https://github.com/yandex-research/heterophilous-graphs)
- [Mã H₂GCN của tác giả](https://github.com/GemsLab/H2GCN)
- [Bài báo H₂GCN](https://arxiv.org/abs/2006.11468)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)

Giấy phép dữ liệu/mã nguồn bên ngoài thuộc tác giả tương ứng; không có mã TensorFlow của tác giả được vendored vào repository này.
