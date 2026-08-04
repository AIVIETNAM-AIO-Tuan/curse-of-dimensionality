"""
Pipeline chính: PCA -> LDA -> KNN, kèm ablation study (so sánh 4 cấu hình:
normal / chỉ PCA / chỉ LDA / PCA->LDA).
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .dim_reduction import apply_pca, apply_lda
from .metrics import METRIC_NAMES, knn_metrics, plot_delta_heatmap

CONFIG_ORDER = ["normal", "pca", "lda", "pca_lda"]


def pca_lda_knn(X, y,
                 n_component_pca=57,
                 n_component_lda=9,
                 n_neighbors=5,
                 test_size=0.3,
                 random_state=42,
                 average="macro",
                 dataset_name="dataset",
                 plot_heatmap=True,
                 verbose=True):
    """
    Pipeline chính: dữ liệu -> apply_pca -> apply_lda -> KNN.
    Kết quả metric chính chỉ lấy của PCA -> LDA.
    Ngoài ra tính thêm 3 biến thể để so sánh delta:
        - không dùng PCA        (chỉ LDA)
        - không dùng LDA        (chỉ PCA)
        - không dùng PCA + LDA  (dữ liệu gốc đã chuẩn hóa)

    Trả về dict:
        {
            "metrics": <metric của pca_lda>,
            "all": {"normal": ..., "pca": ..., "lda": ..., "pca_lda": ...},
            "delta": <DataFrame delta giữa pca_lda và từng cấu hình còn lại>,
        }
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if verbose:
        print(f"Dữ liệu ban đầu : {X.shape} | số class: {len(np.unique(y))}")

    # Bước 0: chia train/test + chuẩn hóa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ============ PIPELINE CHÍNH: PCA -> LDA -> KNN ============
    X_tr_pca, X_te_pca = apply_pca(X_train_s, X_test_s,
                                    n_components=n_component_pca, verbose=verbose)
    X_tr_pl, X_te_pl = apply_lda(X_tr_pca, X_te_pca, y_train,
                                  n_components=n_component_lda, verbose=verbose)
    main = knn_metrics(X_tr_pl, y_train, X_te_pl, y_test, n_neighbors, average)

    if verbose:
        print(f"[KNN k={n_neighbors}] PCA -> LDA -> KNN  (average='{average}')")
        for name in METRIC_NAMES:
            print(f"      {name:<10}: {main[name]:.4f}")

    # ============ CÁC BIẾN THỂ CHỈ DÙNG ĐỂ TÍNH DELTA ============
    # 1. Không dùng PCA -> chỉ LDA trên dữ liệu gốc
    X_tr_l, X_te_l = apply_lda(X_train_s, X_test_s, y_train,
                                n_components=n_component_lda, verbose=False)
    no_pca = knn_metrics(X_tr_l, y_train, X_te_l, y_test, n_neighbors, average)

    # 2. Không dùng LDA -> chỉ PCA
    no_lda = knn_metrics(X_tr_pca, y_train, X_te_pca, y_test, n_neighbors, average)

    # 3. Không dùng PCA + LDA -> dữ liệu gốc đã chuẩn hóa
    no_both = knn_metrics(X_train_s, y_train, X_test_s, y_test, n_neighbors, average)

    # ============ GOM 4 CẤU HÌNH ============
    all_results = {
        "normal":  no_both,   # không PCA, không LDA
        "pca":     no_lda,    # chỉ PCA
        "lda":     no_pca,    # chỉ LDA
        "pca_lda": main,      # PCA -> LDA (pipeline chính)
    }

    # ============ BẢNG DELTA (pca_lda trừ từng cấu hình còn lại) ============
    rows = {
        "vs normal (không PCA + LDA)": no_both,
        "vs pca (không LDA)":          no_lda,
        "vs lda (không PCA)":          no_pca,
    }
    df_delta = pd.DataFrame(
        {m: {label: main[m] - vals[m] for label, vals in rows.items()} for m in METRIC_NAMES}
    )[METRIC_NAMES]

    if verbose:
        print("\n--- DELTA (pca_lda trừ cấu hình khác) ---")
        print(df_delta.round(4))

    if plot_heatmap:
        plot_delta_heatmap(df_delta, dataset_name=dataset_name)

    return {"metrics": main, "all": all_results, "delta": df_delta}


def show_table(res, dataset_name):
    """In và trả về bảng metric (4 cấu hình x 4 metric) của một dataset."""
    df = pd.DataFrame(res["all"]).T.loc[CONFIG_ORDER, METRIC_NAMES]
    print(f"\n--- METRIC ({dataset_name}) ---")
    print(df.round(4))
    return df
