# Ảnh hưởng của các phương pháp giảm chiều trong mô hình KNN

Dự án so sánh bốn cấu hình gồm KNN trên dữ liệu đã chuẩn hóa, PCA-KNN, LDA-KNN và PCA-LDA-KNN trên ba bộ dữ liệu Breast Cancer, LFW People và MNIST. Các cấu hình được đánh giá bằng 3-fold stratified cross-validation với cùng các fold, cùng mô hình KNN và cùng hệ thống metric gồm accuracy, macro-precision, macro-recall và macro-F1.

Phiên bản thực nghiệm bổ sung sử dụng PCA và Linear Discriminant Analysis của scikit-learn. Mọi bước chuẩn hóa và giảm chiều được fit riêng trên training fold nhằm tránh data leakage. Thời gian truy vấn của KNN cũng được đo bằng mili giây trên mỗi mẫu.

## Các file chính

- `Main.ipynb`: notebook của thí nghiệm ban đầu.
- `cv_ablation_results_only.py`: script chạy stratified cross-validation và xuất kết quả.
- `report_table.csv`: bảng mean và standard deviation dùng trong báo cáo.
- `cv_fold_results.csv`: kết quả chi tiết theo từng fold.
- `cv_summary.csv`: thống kê tổng hợp của các pipeline.
- `cv_delta_vs_normal.csv`: chênh lệch so với KNN trên dữ liệu chuẩn hóa.
- `query_time.csv`: thời gian truy vấn và speedup so với baseline.
- `best_pipeline.csv`: pipeline có accuracy trung bình cao nhất trên từng dataset.
- `Technical_Report.pdf`: báo cáo kỹ thuật.
- `requirements.txt`: các thư viện cần thiết.

## Cài đặt

```bash
pip install -r requirements.txt
