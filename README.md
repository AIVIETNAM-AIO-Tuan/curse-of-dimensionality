# Curse of Dimensionality in KNN and the Role of Dimensionality Reduction 🌌

## 📌 Tổng quan dự án
Dự án này tập trung nghiên cứu sự suy giảm hiệu suất của thuật toán K-Nearest Neighbors (KNN) khi số chiều dữ liệu tăng lên ("Lời nguyền số chiều"), và đánh giá xem các phương pháp giảm chiều dữ liệu (như PCA, LDA) có thực sự giúp khôi phục độ chính xác hay không.

Dự án được định hướng bởi 4 câu hỏi nghiên cứu (Research Questions - RQ) cốt lõi:
- **RQ1:** Hiện tượng tập trung khoảng cách (distance concentration - Beyer et al. 1999) có thực sự xảy ra không, và ở số chiều `d` bằng bao nhiêu?
- **RQ2:** Độ chính xác của KNN suy giảm như thế nào theo `d`, và nguyên nhân là do sự tập trung (concentration) hay sự thưa thớt (sparsity)?
- **RQ3:** Thuật toán PCA có thể khôi phục lại bao nhiêu phần trăm hiệu suất, và cần bao nhiêu components?
- **RQ4:** Khi nào thì PCA thất bại (ví dụ: tín hiệu có phương sai thấp nhưng tính phân loại cao; hoặc các đa tạp phi tuyến tính)?

*(Chi tiết xem thêm tại `research/README.md`)*

## 📂 Cấu trúc Repository
Dự án tách biệt rõ ràng giữa **TẠI SAO** (`research/`) và **NHƯ THẾ NÀO** (`models/`, `pipeline/`).

| Folder | Nội dung | README chi tiết |
|---|---|---|
| `data/` | Dữ liệu thô (raw), đã xử lý (processed), và dữ liệu tổng hợp (synthetic). | [data/README.md](data/README.md) |
| `research/` | Nền tảng lý thuyết, literature review và nhật ký thí nghiệm. | [research/README.md](research/README.md) |
| `models/` | Các mô hình Estimators (KNN) và Reducers (PCA, LDA). | [models/README.md](models/README.md) |
| `pipeline/` | Quản lý luồng thí nghiệm: load → preprocess → fit → evaluate. | [pipeline/README.md](pipeline/README.md) |
| `results/` | Output sinh tự động (figures, bảng số liệu, logs). **Không chỉnh sửa thủ công**. | N/A |
| `report/` | Báo cáo tổng kết cuối cùng. | N/A |
| `tests/` | Unit tests cho models và pipelines. | N/A |

## ⚠️ Nguyên tắc cốt lõi & Quy định thực nghiệm
Để đảm bảo tính đúng đắn của thực nghiệm khoa học, mọi thành viên cần tuân thủ các quy định (Domain rules) sau:

**1. Quản lý luồng dữ liệu (Data Flow & Leakage):**
- Thư mục `data/raw/` là **Read-Only**. Mọi biến đổi phải ghi ra `data/processed/`.
- Trật tự bắt buộc để tránh Data Leakage: **Split (chia tập) → Fit scaler (chỉ trên tập Train) → Fit reducer (chỉ trên tập Train)**. Toàn bộ quy trình này nên được đóng gói trong một `sklearn.Pipeline`.

**2. Thiết kế dữ liệu tổng hợp (Synthetic Data):**
- Trình tạo dữ liệu tổng hợp sẽ giữ cố định số lượng *chiều mang thông tin* (informative dimensions) trong khi tăng dần *tổng số chiều*. Điều này giúp cô lập vấn đề "KNN bị ảnh hưởng bởi nhiễu số chiều" thay vì "bài toán trở nên khó hơn".

**3. Quy tắc đo lường & Đánh giá (Evaluation):**
- **Cấm dùng `algorithm="auto"` khi đo thời gian chạy (runtime):** Bắt buộc dùng `algorithm="brute"` cho mọi benchmark runtime của KNN. Nếu để auto, sklearn sẽ tự chuyển đổi thuật toán (`kd_tree`/`ball_tree`) khi `d > 20`, làm sai lệch kết quả đánh giá tốc độ thực sự.
- **Baselines bắt buộc:** Mọi so sánh phải đi kèm baseline `identity` (không giảm chiều) và `random_projection`. Nếu PCA không thắng được random projection (theo bổ đề Johnson–Lindenstrauss), kết luận đúng phải là "giảm chiều có ích", chứ không phải "PCA có ích".
- **Metrics đặc thù:** Ngoài Accuracy/F1, bắt buộc báo cáo:
  - *Intrinsic dimension* (Levina–Bickel MLE) thay vì lấy `X.shape[1]`.
  - *Distance contrast:* `(D_max - D_min) / D_min`.
  - *Hubness skewness.*

## 🚀 Trạng thái hiện tại & Kết quả thực nghiệm (Ablation Study)
Dự án đã hoàn thiện prototype tại [`pipeline/PCA_&_LDA_pipeline.ipynb`](pipeline/PCA_%26_LDA_pipeline.ipynb). 
- Các mô hình giảm chiều `Custom_PCA` và `Custom_LDA` được **hiện thực hoàn toàn bằng NumPy thuần** (eigendecomposition trên ma trận hiệp phương sai / scatter matrix) nhằm phục vụ mục đích nghiên cứu bản chất toán học, tuyệt đối không dùng `sklearn.decomposition`.

Dưới đây là kết quả **Ablation Study** (k=5, average='macro') đánh giá độc lập từng thành phần trong pipeline trên 3 bộ dữ liệu:

### 1. Breast Cancer (d=30, classes=2)
Pipeline giảm chiều: `30 -> PCA(8) -> LDA(1) -> KNN`

| Cấu hình | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Normal (KNN gốc) | 0.9591 | 0.9575 | 0.9544 | 0.9559 |
| Chỉ dùng PCA | 0.9649 | 0.9654 | 0.9590 | 0.9620 |
| Chỉ dùng LDA | 0.9532 | 0.9456 | 0.9563 | 0.9504 |
| **PCA → LDA → KNN** | **0.9825** | **0.9797** | **0.9828** | **0.9812** |

### 2. LFW People (d=1850, classes=62)
Pipeline giảm chiều: `1850 -> PCA(100) -> LDA(61) -> KNN`

| Cấu hình | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Normal (KNN gốc) | 0.2944 | 0.3031 | 0.1531 | 0.1700 |
| Chỉ dùng PCA | 0.2988 | 0.3194 | 0.1747 | 0.1911 |
| Chỉ dùng LDA | 0.1323 | 0.0639 | 0.0606 | 0.0586 |
| **PCA → LDA → KNN** | **0.5050** | **0.5788** | **0.3812** | **0.4194** |

### 3. MNIST (d=784, classes=10)
Pipeline giảm chiều: `784 -> PCA(57) -> LDA(9) -> KNN`

| Cấu hình | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Normal (KNN gốc) | 0.9430 | 0.9433 | 0.9421 | 0.9425 |
| **Chỉ dùng PCA** | **0.9589** | **0.9587** | **0.9585** | **0.9586** |
| Chỉ dùng LDA | 0.9127 | 0.9117 | 0.9113 | 0.9112 |
| PCA → LDA → KNN | 0.9171 | 0.9161 | 0.9159 | 0.9157 |

**💡 Phân tích sơ bộ (Liên hệ RQ4):** Kết quả trên tập MNIST cho thấy `PCA -> LDA` không phải lúc nào cũng mang lại hiệu suất tốt nhất (F1 giảm từ 0.9425 xuống 0.9157 so với Normal, và thấp hơn biến thể chỉ dùng PCA). Đây là minh chứng tuyệt vời để đào sâu vào **RQ4**, đòi hỏi phân tích thêm qua *distance contrast* và các cấu trúc đa tạp phi tuyến tính (nonlinear manifolds) ở các bước nghiên cứu tiếp theo.
