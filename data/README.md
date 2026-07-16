# Data

Chứa toàn bộ dữ liệu dùng cho thí nghiệm: dữ liệu synthetic để kiểm soát số chiều, và dữ liệu thật để kiểm chứng kết luận.

## Cấu trúc hiện tại

```
data/
├── README.md
├── raw/          # Dữ liệu tải về nguyên trạng, không chỉnh sửa
├── processed/    # Dữ liệu sau khi scale / split / encode
├── synthetic/    # Dữ liệu sinh ra để quét số chiều
└── external/     # Nguồn thứ ba không tải tự động được
```

Các thư mục hiện đang rỗng (chỉ có `.gitkeep`).

## Ý nghĩa từng thư mục

**`raw/`** — Dữ liệu tải về giữ nguyên trạng thái gốc. Không chỉnh sửa file ở đây bao giờ, để lúc nào cũng quay lại được điểm xuất phát mà không cần tải lại.

**`processed/`** — Kết quả của bước tiền xử lý: scaling, train/test split, label encoding. Đọc từ `raw/`, ghi ra đây.

**`synthetic/`** — Dữ liệu sinh nhân tạo, là nguồn chính cho câu hỏi nghiên cứu. Ý tưởng thiết kế: giữ **cố định số chiều mang signal** trong khi tăng tổng số chiều, để mọi chiều thêm vào đều là nhiễu thuần túy. Nhờ vậy tách bạch được "KNN chịu chiều thừa đến đâu" khỏi "bài toán có khó hơn không".

**`external/`** — Dữ liệu từ nguồn ngoài phải lấy thủ công. Kèm ghi chú nguồn gốc khi thêm vào.

## Dataset dự kiến

| Dataset | n | d | classes | Vì sao chọn |
|---|---|---|---|---|
| Synthetic | tùy chỉnh | quét | 2 | Kiểm soát hoàn toàn số chiều signal vs nhiễu |
| MNIST | 70,000 | 784 | 10 | Số chiều thật (~12-14) thấp hơn nhiều so với 784 cột |
| Fashion-MNIST | 70,000 | 784 | 10 | Cùng shape MNIST nhưng khó hơn — kiểm chứng kết luận |
| Isolet | 7,797 | 617 | 26 | d cao, n nhỏ — chế độ khắc nghiệt nhất cho KNN |

## Lưu ý quan trọng khi xử lý

Thứ tự bắt buộc: **split trước → fit scaler chỉ trên train → fit reducer chỉ trên train**.

Fit PCA trên toàn bộ dữ liệu trước khi split là dạng leakage phổ biến nhất trong loại thí nghiệm này — test set góp phần định nghĩa các trục chính, khiến accuracy báo cáo cao hơn thực tế.

Scaling là bắt buộc với KNN vì khoảng cách Euclid không bất biến với đơn vị đo.

## Git

File dữ liệu (`.npz`, `.csv`, `.gz`, `.pkl`) không commit. Chỉ commit `.gitkeep` và README.
