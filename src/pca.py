"""
Cài đặt PCA thủ công (chỉ dùng NumPy), tách khỏi notebook để có thể tái sử dụng
ở nhiều nơi khác nhau (Main.ipynb, script khác, unit test, ...).
"""

import numpy as np


class Custom_PCA:
    """
    PCA tự hiện thực theo công thức toán học, chỉ dùng NumPy.
        1. Căn giữa dữ liệu:      Xc = X - mean
        2. Ma trận hiệp biến:     C  = (Xc^T @ Xc) / (n - 1)
        3. Trị riêng / vectơ riêng của C
        4. Sắp xếp trị riêng giảm dần, lấy k vectơ riêng đầu -> W
        5. Chiếu dữ liệu:         Z  = Xc @ W
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None            # W: (n_features, k)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        # 1. Căn giữa dữ liệu theo mean của tập train
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_

        # 2. Ma trận hiệp phương sai
        n_samples = Xc.shape[0]
        cov_matrix = (Xc.T @ Xc) / (n_samples - 1)

        # 3. Trị riêng & vectơ riêng (eigh cho ma trận đối xứng -> kết quả thực)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sắp xếp giảm dần theo trị riêng
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 5. Lưu k thành phần chính đầu tiên
        k = self.n_components
        self.components_ = eigenvectors[:, :k]
        self.explained_variance_ = eigenvalues[:k]

        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
