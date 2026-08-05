"""
Các hàm tiện ích để áp dụng Custom_PCA / Custom_LDA lên dữ liệu train/test.
"""

import numpy as np

from .pca import Custom_PCA
from .lda import Custom_LDA


def apply_pca(X_train, X_test, n_components, verbose=True):
    """
    Giảm chiều bằng Custom_PCA (thủ công).
    Trả về: (X_train_pca, X_test_pca) — dữ liệu sau giảm chiều.
    """
    dim_in = np.asarray(X_train).shape[1]

    pca = Custom_PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    if verbose:
        print(f"[PCA] số chiều: {dim_in} -> {X_train_pca.shape[1]}")
        print(f"      X_train: {np.asarray(X_train).shape} -> {X_train_pca.shape}")
        print(f"      X_test : {np.asarray(X_test).shape} -> {X_test_pca.shape}")
        print(f"      Phương sai giữ lại: {pca.explained_variance_ratio_.sum():.4f}")

    return X_train_pca, X_test_pca


def apply_lda(X_train, X_test, y_train, n_components, verbose=True):
    """
    Giảm chiều bằng Custom_LDA (thủ công).
    Đầu vào có thể là dữ liệu gốc HOẶC dữ liệu đã qua apply_pca.
    Trả về: (X_train_lda, X_test_lda) — dữ liệu sau giảm chiều.
    """
    dim_in = np.asarray(X_train).shape[1]
    n_classes = len(np.unique(np.asarray(y_train)))

    lda = Custom_LDA(n_components=n_components)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)

    if verbose:
        if lda.n_components < n_components:
            print(f"[LDA] n_components={n_components} vượt giới hạn "
                  f"(số class - 1 = {n_classes - 1}) -> dùng {lda.n_components}")
        print(f"[LDA] số chiều: {dim_in} -> {X_train_lda.shape[1]} (số class = {n_classes})")
        print(f"      X_train: {np.asarray(X_train).shape} -> {X_train_lda.shape}")
        print(f"      X_test : {np.asarray(X_test).shape} -> {X_test_lda.shape}")

    return X_train_lda, X_test_lda
