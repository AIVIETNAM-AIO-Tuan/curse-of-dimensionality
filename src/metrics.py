"""
Các hàm tính metric (accuracy/precision/recall/f1) cho KNN và vẽ heatmap delta
giữa các cấu hình trong ablation study.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def knn_metrics(X_tr, y_tr, X_te, y_te, n_neighbors=5, average="macro"):
    """Train KNN rồi trả về dict accuracy / precision / recall / f1."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_tr, y_tr)
    y_pred = knn.predict(X_te)
    return {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average=average, zero_division=0),
        "recall":    recall_score(y_te, y_pred, average=average, zero_division=0),
        "f1":        f1_score(y_te, y_pred, average=average, zero_division=0),
    }


def plot_delta_heatmap(df_delta, dataset_name="dataset"):
    """Heatmap delta: PCA->LDA so với từng biến thể (dương = PCA->LDA tốt hơn)."""
    vmax = np.abs(df_delta.values).max()
    vmax = 0.01 if vmax == 0 else vmax

    plt.figure(figsize=(7.5, 3.4))
    sns.heatmap(df_delta, annot=True, fmt="+.4f", cmap="RdBu_r",
                center=0, vmin=-vmax, vmax=vmax, linewidths=0.5,
                cbar_kws={"label": "delta (PCA->LDA - biến thể)"})
    plt.title(f"Delta metric của PCA->LDA->KNN | {dataset_name}\n"
              f"(đỏ = PCA->LDA tốt hơn, xanh = tệ hơn)")
    plt.xlabel("Metric")
    plt.ylabel("So với")
    plt.tight_layout()
    plt.show()
