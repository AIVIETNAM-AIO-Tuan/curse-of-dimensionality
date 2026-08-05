"""
Package src: các thành phần tái sử dụng của pipeline PCA -> LDA -> KNN.

- src.pca            : Custom_PCA
- src.lda             : Custom_LDA
- src.dim_reduction   : apply_pca, apply_lda
- src.metrics         : METRIC_NAMES, knn_metrics, plot_delta_heatmap
- src.pipeline        : pca_lda_knn, show_table, CONFIG_ORDER
"""

from .pca import Custom_PCA
from .lda import Custom_LDA
from .dim_reduction import apply_pca, apply_lda
from .metrics import METRIC_NAMES, knn_metrics, plot_delta_heatmap
from .pipeline import pca_lda_knn, show_table, CONFIG_ORDER

__all__ = [
    "Custom_PCA",
    "Custom_LDA",
    "apply_pca",
    "apply_lda",
    "METRIC_NAMES",
    "knn_metrics",
    "plot_delta_heatmap",
    "pca_lda_knn",
    "show_table",
    "CONFIG_ORDER",
]
