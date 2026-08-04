# PCA → LDA → KNN: Dimensionality Reduction Pipeline & Ablation Study

Cài đặt thủ công (chỉ dùng NumPy) hai phương pháp giảm chiều **PCA** và **LDA**,
kết hợp thành pipeline `PCA -> LDA -> KNN`, và thực hiện ablation study để đánh
giá đóng góp riêng của từng bước giảm chiều trên 3 bộ dữ liệu: **Breast Cancer**,
**LFW People**, **MNIST**.

## Cấu trúc thư mục

```
.
├── README.md                  # File này
├── requirements.txt            # Danh sách thư viện Python cần cài
├── Technical Report.pdf       # Báo cáo kỹ thuật đầy đủ (cơ sở lý thuyết, phương
│                               # pháp, kết quả thực nghiệm, phân tích ablation)
├── Main.ipynb                 # Notebook chạy thực nghiệm: chỉ import từ src/,
│                               # không chứa logic nghiệp vụ
└── src/                       # Các thành phần tái sử dụng (logic chính)
    ├── __init__.py            # Export public API của package
    ├── pca.py                 # Custom_PCA — PCA tự cài đặt bằng NumPy
    ├── lda.py                 # Custom_LDA — LDA (Fisher) tự cài đặt bằng NumPy
    ├── dim_reduction.py       # apply_pca, apply_lda — hàm tiện ích áp dụng
    │                          #   Custom_PCA / Custom_LDA lên train/test
    ├── metrics.py             # knn_metrics, plot_delta_heatmap — tính metric
    │                          #   và trực quan hóa kết quả ablation
    └── pipeline.py            # pca_lda_knn, show_table — pipeline chính
                                #   (bao gồm 4 cấu hình ablation: normal / pca /
                                #   lda / pca_lda)
```

**Nguyên tắc tổ chức:** mọi logic có thể tái sử dụng (class, hàm xử lý dữ liệu,
hàm tính metric, pipeline) được tách vào `src/` dưới dạng module Python thuần.
`Main.ipynb` chỉ đóng vai trò gọi các hàm trong `src/` để chạy thực nghiệm và
hiển thị kết quả, không định nghĩa logic trực tiếp trong notebook.

## Yêu cầu môi trường

```bash
pip install -r requirements.txt
```

(Python ≥ 3.9. Không cần cài `nbformat` để chạy notebook, chỉ cần Jupyter/JupyterLab.)

## Cách chạy

1. Mở `Main.ipynb` bằng Jupyter/JupyterLab hoặc VS Code.
2. Chạy tuần tự các cell — notebook sẽ:
   - Tải 3 bộ dữ liệu (`load_breast_cancer`, `fetch_lfw_people`, `fetch_openml('mnist_784')`).
   - Gọi `pca_lda_knn(...)` từ `src.pipeline` cho từng bộ dữ liệu, thực hiện
     ablation với 4 cấu hình: `normal` (không giảm chiều), `pca` (chỉ PCA),
     `lda` (chỉ LDA), `pca_lda` (PCA → LDA).
   - In bảng metric (accuracy, precision, recall, f1) và heatmap delta cho
     từng bộ dữ liệu.
   - Gộp kết quả 3 bộ dữ liệu × 4 cấu hình thành một bảng tổng hợp `df_summary`.

Cũng có thể dùng trực tiếp các thành phần trong `src/` ở script hoặc notebook khác:

```python
from src import pca_lda_knn, show_table, Custom_PCA, Custom_LDA

res = pca_lda_knn(X, y, n_component_pca=8, n_component_lda=1,
                   dataset_name="My Dataset")
show_table(res, "My Dataset")
```

## Tài liệu tham khảo chi tiết

Toàn bộ cơ sở lý thuyết (công thức PCA/LDA), mô tả phương pháp, thiết lập
thực nghiệm, kết quả và phân tích ablation được trình bày đầy đủ trong
[`Technical Report.pdf`](./Technical%20Report.pdf).