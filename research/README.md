# Research Documents

Nền tảng lý thuyết, khảo sát tài liệu và nhật ký thí nghiệm. Đây là nơi trả lời **tại sao** — `models/` và `pipeline/` trả lời **như thế nào**.

## Cấu trúc hiện tại

```
research/
├── README.md
├── literature/    # Mỗi paper đã đọc một file ghi chú
├── notes/         # Dẫn giải lý thuyết, chứng minh, phân tích
└── experiments/   # Nhật ký thí nghiệm: giả thuyết → thiết kế → kết quả
```

Các thư mục hiện đang rỗng.

## Ý nghĩa từng thư mục

**`literature/`** — Mỗi paper một file: paper đặt câu hỏi gì, phương pháp, kết quả chính, phần liên quan tới repo này, và **giới hạn** của paper.

**`notes/`** — Ghi chú lý thuyết tự viết: dẫn giải công thức, chứng minh, phân tích khái niệm. Là phần cầu nối giữa paper và code.

**`experiments/`** — Nhật ký từng thí nghiệm. Điểm mấu chốt: **ghi giả thuyết trước khi chạy**. Sau khi nhìn thấy đồ thị, mọi kết quả đều trông như đã được dự đoán từ đầu.

## Bản đồ câu hỏi nghiên cứu

Câu hỏi lớn tách thành bốn câu hỏi con kiểm định được:

**RQ1 — Distance concentration có thật sự xảy ra không, từ chiều nào?**
Đo tỉ số tương phản khoảng cách theo `d`. Lý thuyết nói "khi `d → ∞`" nhưng thực nghiệm cần con số cụ thể.

**RQ2 — Accuracy KNN suy giảm theo `d` ra sao?**
Cần phân biệt suy giảm do concentration với suy giảm do dữ liệu thưa — hai cơ chế khác nhau, cùng biểu hiện.

**RQ3 — PCA khôi phục được bao nhiêu, ở bao nhiêu components?**
Quét `n_components`, đối chiếu explained variance.

**RQ4 — Khi nào PCA thất bại?**
PCA tìm subspace tuyến tính phương sai lớn và không biết gì về nhãn. Hai chế độ thất bại: (a) signal phương sai thấp nhưng phân biệt tốt → PCA vứt đúng thứ cần giữ; (b) signal nằm trên manifold phi tuyến.

RQ4 dễ bị bỏ qua nhất và cũng là phần phân biệt báo cáo mô tả với báo cáo nghiên cứu. Trả lời "có, PCA giúp" mà không nêu điều kiện là kết luận sai.

## Tóm tắt lý thuyết nền

**Distance concentration.** Beyer et al. (1999): khi phương sai tương đối của khoảng cách triệt tiêu theo `d`, thì tỉ số `(D_max − D_min) / D_min → 0`. Mọi điểm trở nên gần như cách đều nhau, "nearest neighbor" mất ý nghĩa.

Điều kiện này thỏa mãn khi các chiều độc lập, và **không** thỏa mãn khi các chiều tương quan mạnh. Dữ liệu thật hầu như luôn tương quan — đó là lý do KNN vẫn chạy được trên MNIST 784 chiều. Curse of dimensionality là hàm của **intrinsic dimension**, không phải số cột trong ma trận.

**Hubness.** Ở chiều cao, một số ít điểm trở thành neighbor của rất nhiều điểm khác (Radovanović et al., 2010). Hub gán nhãn sai làm hỏng dự đoán lan truyền. Là hệ quả riêng biệt của concentration, nên đo riêng.

**Vì sao PCA giúp.** Chiều nhiễu phương sai thấp bị loại → tỉ số signal/noise trong khoảng cách tăng; `d` giảm → contrast phục hồi; runtime giảm tuyến tính theo `d`. Cả ba đều có điều kiện ở giả thiết "signal nằm trong subspace tuyến tính chiều thấp".

## Tài liệu tham khảo chính

- Beyer, K. et al. (1999). *When Is "Nearest Neighbor" Meaningful?* ICDT.
- Aggarwal, C. C. et al. (2001). *On the Surprising Behavior of Distance Metrics in High Dimensional Space.* ICDT.
- Radovanović, M. et al. (2010). *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data.* JMLR.
- Jolliffe, I. T. (2002). *Principal Component Analysis.* Springer.
- Levina, E. & Bickel, P. (2004). *Maximum Likelihood Estimation of Intrinsic Dimension.* NIPS.
