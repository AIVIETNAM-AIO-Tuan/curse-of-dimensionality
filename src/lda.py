"""
Cài đặt LDA (Fisher's Linear Discriminant Analysis) thủ công, chỉ dùng NumPy.
Tách khỏi notebook để có thể tái sử dụng như một module độc lập.
"""

import numpy as np


class Custom_LDA:
    """
    LDA (Fisher) tự hiện thực theo công thức toán học, chỉ dùng NumPy.
        1. mean tổng thể và mean từng lớp
        2. S_W = sum_c (X_c - m_c)^T (X_c - m_c)          (within-class scatter)
        3. S_B = sum_c n_c (m_c - m)(m_c - m)^T           (between-class scatter)
        4. Trị riêng / vectơ riêng của pinv(S_W) @ S_B
        5. Sắp xếp trị riêng giảm dần, lấy k vectơ riêng đầu -> W
        6. Chiếu dữ liệu: Z = X @ W      (k <= n_classes - 1)
    """

    def __init__(self, n_components):
        self.n_components = n_components
        self.classes_ = None
        self.components_ = None            # W: (n_features, k)
        self.eigenvalues_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_features = X.shape[1]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # LDA chỉ tách được tối đa (số class - 1) chiều
        k = min(self.n_components, n_classes - 1, n_features)
        self.n_components = k

        # 1. Mean tổng thể & mean từng lớp
        mean_overall = np.mean(X, axis=0)

        # 2. Within-class scatter S_W
        S_W = np.zeros((n_features, n_features))
        mean_vectors = {}
        for c in self.classes_:
            X_c = X[y == c]
            m_c = np.mean(X_c, axis=0)
            mean_vectors[c] = m_c
            diff = X_c - m_c
            S_W += diff.T @ diff

        # 3. Between-class scatter S_B
        S_B = np.zeros((n_features, n_features))
        for c in self.classes_:
            n_c = X[y == c].shape[0]
            mean_diff = (mean_vectors[c] - mean_overall).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 4. Trị riêng của pinv(S_W) @ S_B  (pinv để tránh S_W suy biến)
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # 5. Sắp xếp giảm dần, lấy k vectơ riêng đầu
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components_ = eigenvectors[:, :k]
        self.eigenvalues_ = eigenvalues[:k]

        # Chuẩn hóa độ dài vectơ chiếu cho ổn định số học
        norms = np.linalg.norm(self.components_, axis=0)
        norms[norms == 0] = 1.0
        self.components_ = self.components_ / norms
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.components_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
