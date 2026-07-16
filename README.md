# Curse of Dimensionality in KNN and the Role of Dimensionality Reduction

**Câu hỏi nghiên cứu:** Khi số chiều dữ liệu tăng lên, KNN suy giảm như thế nào và dimensionality reduction có giúp cải thiện không?

**Outcome dự kiến:** synthetic experiment, real dataset experiment, PCA comparison, accuracy/runtime analysis và visualization.

## Cấu trúc repo

```
knn-curse-of-dimensionality/
├── README.md          # File này — tổng quan toàn dự án
├── data/              # Dữ liệu: synthetic + real dataset
├── research/          # Tài liệu nghiên cứu: lý thuyết, paper, nhật ký thí nghiệm
├── models/            # KNN và các phương pháp dimensionality reduction
├── pipeline/          # Luồng chạy thí nghiệm: load → preprocess → fit → evaluate
├── results/           # Output sinh tự động: figures, bảng số liệu, logs
├── report/            # Báo cáo cuối (viết sau)
└── tests/             # Test cho model và pipeline
```

Mỗi folder chính có README riêng mô tả chi tiết nội dung bên trong:

| Folder | Nội dung | README |
|---|---|---|
| `data/` | Dataset thô, đã xử lý, synthetic | [data/README.md](data/README.md) |
| `research/` | Nền tảng lý thuyết và literature | [research/README.md](research/README.md) |
| `models/` | Estimators và reducers | [models/README.md](models/README.md) |
| `pipeline/` | Orchestration thí nghiệm | [pipeline/README.md](pipeline/README.md) |

## Nguyên tắc tổ chức

- `data/raw/` không bao giờ bị chỉnh sửa — mọi biến đổi đọc từ đây và ghi ra `data/processed/`.
- `results/` và `report/` là **output**, không phải input. Mọi thứ trong `results/` phải tái tạo được từ code + config.
- `research/` trả lời **tại sao**, `models/` + `pipeline/` trả lời **như thế nào**.

## Trạng thái

Repo mới khởi tạo — hiện chỉ có cấu trúc thư mục và tài liệu. Code chưa được viết.
