# PCA → LDA → KNN: Dimensionality Reduction Pipeline & Ablation Study

Dự án đánh giá ảnh hưởng của PCA và LDA đối với KNN thông qua bốn cấu hình:
`normal`, `pca`, `lda` và `pca_lda`. Thí nghiệm được thực hiện trên Breast
Cancer, LFW People và MNIST bằng accuracy, macro-precision, macro-recall,
macro-F1 và thời gian truy vấn của KNN.

PCA và LDA được cài đặt thủ công bằng NumPy. `scikit-learn` được sử dụng để tải
dữ liệu, chia train/test, chuẩn hóa, huấn luyện KNN và tính metric.

## Cấu trúc repository

| Đường dẫn | Vai trò |
|---|---|
| `Main.ipynb` | Tải dữ liệu, gọi pipeline và hiển thị bảng/heatmap |
| `query_time_main10.py` | Đo riêng thời gian `KNN.predict` cho bốn cấu hình |
| `results/query_time_main10.csv` | Kết quả query time dùng trong báo cáo |
| `Technical_Report.pdf` | Báo cáo kỹ thuật |
| `requirements.txt` | Danh sách thư viện Python |
| `src/pca.py` | Cài đặt `Custom_PCA` |
| `src/lda.py` | Cài đặt `Custom_LDA` |
| `src/dim_reduction.py` | Các hàm `apply_pca` và `apply_lda` |
| `src/metrics.py` | Metric phân loại và heatmap delta |
| `src/pipeline.py` | Pipeline và bốn cấu hình ablation |

Mọi logic có thể tái sử dụng nằm trong `src/`. Notebook và script ở thư mục
gốc chỉ điều phối thí nghiệm.

## Cài đặt

Yêu cầu Python 3.9 trở lên.

```bash
python -m pip install -r requirements.txt
```

## Chạy thí nghiệm phân loại

Mở `Main.ipynb` bằng Jupyter, JupyterLab, VS Code hoặc Google Colab và chạy tuần
tự toàn bộ cell. Notebook sẽ:

1. tải Breast Cancer, LFW People và MNIST;
2. gọi `pca_lda_knn(...)` từ `src.pipeline`;
3. đánh giá `normal`, `pca`, `lda` và `pca_lda`;
4. in accuracy, macro-precision, macro-recall và macro-F1;
5. vẽ heatmap delta và tạo bảng tổng hợp.

Các thiết lập chính được giữ cố định: `test_size=0.3`, `random_state=42` và
KNN với `k=5`.

## Chạy phép đo query time

Từ thư mục gốc của repository:

```bash
python query_time_main10.py --output results/query_time_main10.csv
```

Có thể chạy riêng một hoặc nhiều bộ dữ liệu:

```bash
python query_time_main10.py --datasets breast_cancer lfw
```

Thiết lập mặc định sử dụng tối đa 200 mẫu test, một lượt warm-up và năm lần
đo. Giá trị báo cáo là trung vị mili giây trên mỗi mẫu. Phép đo chỉ bao gồm
`KNeighborsClassifier.predict` sau khi dữ liệu đã được chuẩn hóa và biến đổi;
không bao gồm thời gian fit, StandardScaler, PCA hoặc LDA. Vì thời gian tuyệt
đối phụ thuộc phần cứng và tải hệ thống, kết quả chủ yếu dùng để so sánh tương
đối giữa các cấu hình trong cùng một lần chạy.

Trong Google Colab hoặc Jupyter, chạy script bằng lệnh shell:

```python
!python query_time_main10.py --output results/query_time_main10.csv
```

## Báo cáo

Cơ sở lý thuyết, thiết lập thực nghiệm, kết quả phân loại, query time, thảo
luận và hạn chế được trình bày trong
[`Technical_Report.pdf`](./Technical_Report.pdf).
