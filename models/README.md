# Models

Chứa KNN classifier và các phương pháp dimensionality reduction. Mục tiêu: đưa tất cả về một API thống nhất để pipeline hoán đổi chúng qua config mà không cần sửa code.

## Cấu trúc hiện tại

```
models/
├── README.md
└── artifacts/    # Model đã fit, sinh tự động — không commit
```

Code chưa được viết. Dự kiến gồm:

| File | Vai trò |
|---|---|
| `knn.py` | KNN classifier + các biến thể metric / weighting |
| `reducers.py` | Các phương pháp giảm chiều |
| `metrics.py` | Metrics riêng của bài toán: distance contrast, hubness, intrinsic dimension |
| `registry.py` | Ánh xạ tên trong config → constructor |

## KNN — các tham số cần quét

| Tham số | Lựa chọn | Ghi chú |
|---|---|---|
| `n_neighbors` | int | `k`. Tune bằng CV trên train, không bao giờ trên test |
| `metric` | euclidean, manhattan, chebyshev | L1 chịu chiều cao tốt hơn L2 (Aggarwal 2001) — biến đáng quét |
| `weights` | uniform, distance | `distance` kỳ vọng kém dần khi `d` tăng: mọi khoảng cách bằng nhau thì trọng số cũng bằng nhau |
| `algorithm` | brute, kd_tree, ball_tree | Xem cảnh báo bên dưới |

**Cảnh báo khi đo runtime:** `kd_tree` và `ball_tree` tự suy biến về brute force khi `d` lớn (từ khoảng `d > 20`). Nếu để `algorithm="auto"`, sklearn sẽ **âm thầm đổi thuật toán** giữa chừng khi quét `d`, và đường runtime vẽ được sẽ là hình dạng của heuristic chọn thuật toán chứ không phải của curse of dimensionality. Khi so runtime phải pin cứng `brute`.

## Dimensionality reduction — các phương pháp dự kiến

| Tên | Vai trò trong thiết kế |
|---|---|
| `identity` | **Baseline bắt buộc.** Không có nó thì "PCA giúp" là câu không có nghĩa |
| `pca` | Phương pháp chính đang xét |
| `random_projection` | Baseline đối chứng: giảm chiều *không* nhìn dữ liệu |
| `select_k_best` | Có dùng nhãn — cận trên cho phần "biết nhãn thì tốt hơn bao nhiêu" |
| `truncated_svd` | Cho dữ liệu thưa (không center) |

`random_projection` là baseline quan trọng nhất và cũng dễ bị bỏ sót nhất. Johnson–Lindenstrauss đảm bảo phép chiếu ngẫu nhiên giữ gần đúng khoảng cách theo cặp. Nếu nó bám sát PCA thì kết luận đúng phải là "giảm chiều giúp", không phải "PCA giúp".

## Metrics riêng của bài toán

Ngoài accuracy / F1 tiêu chuẩn, cần đo:

- **Distance contrast** — `(D_max − D_min) / D_min`. Đại lượng trung tâm của RQ1; tiến về 0 nghĩa là "hàng xóm gần" hết ý nghĩa.
- **Relative variance** — chính là đại lượng trong điều kiện của định lý Beyer.
- **Hubness skewness** — độ lệch phân bố k-occurrence. Dương lớn = có hub.
- **Intrinsic dimension** (ước lượng MLE, Levina–Bickel) — đại lượng thật sự dự báo hiệu năng KNN, không phải `X.shape[1]`. Nên là con số đầu tiên báo cáo cho mỗi dataset.
