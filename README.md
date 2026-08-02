# Ảnh hưởng của các phương pháp giảm chiều trong mô hình KNN

Dự án đánh giá hiệu năng của mô hình KNN kết hợp hai phương pháp giảm chiều **PCA** (Principal Component Analysis) và **LDA** (Linear Discriminant Analysis), sau đó kết hợp thành pipeline `PCA → LDA → KNN` để phân loại. Ngoài ra, dự án thực hiện **ablation study** nhằm đánh giá đóng góp riêng của từng bước giảm chiều, so sánh trên 3 bộ dữ liệu: **Breast Cancer**, **LFW People**, và **MNIST**.

## Cấu trúc thư mục

```
.
├── Main.ipynb              # Toàn bộ code: cài đặt PCA/LDA thủ công, pipeline, ablation study
├── TechnicalReport.pdf     # Báo cáo kỹ thuật (technical report)
└── README.md
```

## Nội dung `Main.ipynb`

**Phần 1 — Pipeline chính**
- Cài đặt `Custom_PCA` và `Custom_LDA` thủ công bằng NumPy (không dùng `sklearn.decomposition`)
- Hàm `apply_pca`, `apply_lda` để giảm chiều dữ liệu train/test
- Pipeline `pca_lda_knn`: chuẩn hóa → PCA → LDA → phân loại KNN
- So sánh accuracy với mô hình KNN gốc (không giảm chiều) trên 3 bộ dữ liệu

**Phần 2 — Ablation Study**
- Bổ sung các metric: accuracy, precision, recall, f1-score
- So sánh 4 cấu hình: `normal` (không PCA+LDA), `pca` (chỉ PCA), `lda` (chỉ LDA), `pca_lda` (pipeline đầy đủ)
- Vẽ heatmap delta so sánh giữa các cấu hình
- Bảng tổng hợp kết quả trên cả 3 bộ dữ liệu

## Yêu cầu môi trường

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Cách chạy

Mở và chạy tuần tự `Main.ipynb` (ví dụ bằng Jupyter Notebook hoặc Google Colab). Lưu ý bộ dữ liệu LFW People và MNIST sẽ được tải tự động qua `sklearn.datasets` (`fetch_lfw_people`, `fetch_openml`) và có thể mất vài phút cho lần chạy đầu tiên.

## Báo cáo

Chi tiết về cơ sở toán học, phương pháp luận và phân tích kết quả được trình bày đầy đủ trong `report.pdf`.
