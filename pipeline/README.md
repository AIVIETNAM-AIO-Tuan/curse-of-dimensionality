# Pipeline

Orchestration của thí nghiệm: nối dữ liệu, model và metrics thành một luồng chạy được bằng một lệnh, tái tạo được, và không leakage.

## Cấu trúc hiện tại

```
pipeline/
├── README.md
└── steps/    # Các bước rời của luồng
```

Code chưa được viết. Dự kiến gồm:

| File | Vai trò |
|---|---|
| `data_loader.py` | Tải dataset thật + sinh dữ liệu synthetic |
| `preprocess.py` | Split, scaling, đóng gói thành sklearn Pipeline |
| `run_experiment.py` | Entry point: đọc config, chạy sweep, ghi kết quả |
| `evaluate.py` | Tính accuracy, runtime và các metrics riêng |
| `visualize.py` | Sinh figures từ bảng kết quả |
| `config.py` | Đọc và validate file config |

## Luồng chạy

```
load → split → scale → reduce → fit KNN → evaluate → ghi results/
```

Toàn bộ chuỗi scale → reduce → fit được đóng gói trong một `sklearn.Pipeline`. Lý do: leakage bị chặn về mặt cấu trúc, không phụ thuộc vào việc người viết có nhớ fit đúng thứ tự hay không.

## Thí nghiệm dự kiến

**1. Synthetic sweep** — quét `d ∈ {2, 5, 10, 20, 50, 100, 200, 500, 1000}` với số chiều informative cố định. Đo accuracy và distance contrast theo `d`.

**2. Real dataset** — lặp lại trên MNIST / Fashion-MNIST / Isolet, nơi cấu trúc intrinsic dimension không do ta kiểm soát.

**3. PCA comparison** — quét `n_components`, đối chiếu với explained variance để tìm điểm cân bằng.

**4. Accuracy / runtime analysis** — tách biệt hai lợi ích của reduction: chất lượng dự đoán và chi phí tính toán. Chúng không nhất thiết đạt đỉnh ở cùng một `n_components`.

**5. Visualization** — đường accuracy-vs-dimension, histogram phân bố khoảng cách theo `d`, scree plot, scatter 2D sau chiếu.

## Config

Thí nghiệm mô tả bằng file YAML, không hardcode trong code. Đổi thí nghiệm = đổi config, không sửa code.

Config dự kiến trong `configs/`:
- `synthetic_sweep.yaml`
- `real_mnist.yaml`
- `pca_comparison.yaml`
- `runtime_analysis.yaml`

## Reproducibility

- Seed cố định trong config, truyền xuống numpy và mọi estimator.
- Split cố định bằng `stratify` + seed.
- Runtime đo bằng `time.perf_counter`, trung bình qua nhiều lần chạy.
- Mỗi lần chạy ghi ra thư mục theo `run_id` trong `results/`, **kèm bản copy của config đã dùng** — để mọi số liệu trong báo cáo truy vết được về đúng config sinh ra nó.
