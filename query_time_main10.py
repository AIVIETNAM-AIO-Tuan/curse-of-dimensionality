"""Đo thời gian truy vấn KNN cho bốn cấu hình của thí nghiệm chính.

Phép đo chỉ bao gồm ``KNeighborsClassifier.predict`` sau khi dữ liệu đã được
chuẩn hóa và biến đổi. Thời gian fit, StandardScaler, PCA và LDA không được tính.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people, fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from src.dim_reduction import apply_lda, apply_pca


DATASET_CONFIGS = {
    "breast_cancer": {
        "display_name": "Breast Cancer",
        "n_component_pca": 8,
        "n_component_lda": 1,
    },
    "lfw": {
        "display_name": "LFW People",
        "n_component_pca": 100,
        "n_component_lda": 61,
    },
    "mnist": {
        "display_name": "MNIST",
        "n_component_pca": 57,
        "n_component_lda": 9,
    },
}

CONFIG_ORDER = ("normal", "pca", "lda", "pca_lda")


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Tải một trong ba bộ dữ liệu bằng đúng thiết lập của notebook chính."""
    if name == "breast_cancer":
        dataset = load_breast_cancer()
        return dataset.data, dataset.target
    if name == "lfw":
        dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
        return dataset.data, dataset.target
    if name == "mnist":
        try:
            dataset = fetch_openml(
                "mnist_784", version=1, parser="auto", as_frame=False
            )
        except TypeError:
            dataset = fetch_openml("mnist_784", version=1, as_frame=False)
        return dataset.data, dataset.target.astype("int64")
    raise ValueError(f"Dataset không được hỗ trợ: {name}")


def prepare_representations(
    X: np.ndarray,
    y: np.ndarray,
    n_component_pca: int,
    n_component_lda: int,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Tạo train/test representation cho bốn cấu hình ablation."""
    X_train, X_test, y_train, _ = train_test_split(
        np.asarray(X, dtype=np.float64),
        np.asarray(y),
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_pca, X_test_pca = apply_pca(
        X_train_s,
        X_test_s,
        n_components=n_component_pca,
        verbose=False,
    )
    X_train_lda, X_test_lda = apply_lda(
        X_train_s,
        X_test_s,
        y_train,
        n_components=n_component_lda,
        verbose=False,
    )
    X_train_pca_lda, X_test_pca_lda = apply_lda(
        X_train_pca,
        X_test_pca,
        y_train,
        n_components=n_component_lda,
        verbose=False,
    )

    representations = {
        "normal": (X_train_s, X_test_s),
        "pca": (X_train_pca, X_test_pca),
        "lda": (X_train_lda, X_test_lda),
        "pca_lda": (X_train_pca_lda, X_test_pca_lda),
    }
    return representations, y_train


def measure_predict_time(
    model: KNeighborsClassifier,
    X_query: np.ndarray,
    warmups: int,
    repeats: int,
) -> tuple[float, list[float]]:
    """Trả về trung vị ms/sample và toàn bộ các lần đo."""
    for _ in range(warmups):
        model.predict(X_query)

    elapsed_ms_per_sample = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        model.predict(X_query)
        elapsed = time.perf_counter() - start
        elapsed_ms_per_sample.append(elapsed * 1000.0 / len(X_query))

    return float(np.median(elapsed_ms_per_sample)), elapsed_ms_per_sample


def benchmark_dataset(
    name: str,
    max_query_samples: int,
    warmups: int,
    repeats: int,
    n_neighbors: int = 5,
) -> list[dict[str, float | int | str]]:
    """Đo bốn cấu hình trên một dataset và trả về các dòng kết quả."""
    config = DATASET_CONFIGS[name]
    X, y = load_dataset(name)
    representations, y_train = prepare_representations(
        X,
        y,
        n_component_pca=config["n_component_pca"],
        n_component_lda=config["n_component_lda"],
    )

    rows = []
    for configuration in CONFIG_ORDER:
        X_train, X_test = representations[configuration]
        X_query = X_test[: min(max_query_samples, len(X_test))]

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        median_ms, repeated_values = measure_predict_time(
            model,
            X_query,
            warmups=warmups,
            repeats=repeats,
        )

        row: dict[str, float | int | str] = {
            "dataset": config["display_name"],
            "config": configuration,
            "knn_dimension": X_train.shape[1],
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "timed_query_samples": len(X_query),
            "warmups": warmups,
            "repeats": repeats,
            "median_ms_per_sample": median_ms,
        }
        for index, value in enumerate(repeated_values, start=1):
            row[f"repeat_{index}_ms_per_sample"] = value
        rows.append(row)

    normal_time = float(rows[0]["median_ms_per_sample"])
    for row in rows:
        row["speedup_vs_normal"] = normal_time / float(
            row["median_ms_per_sample"]
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đo thời gian KNN.predict cho thí nghiệm PCA/LDA/KNN."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS),
    )
    parser.add_argument(
        "--output",
        default="results/query_time_main10.csv",
        help="Đường dẫn CSV đầu ra.",
    )
    parser.add_argument("--max-query-samples", type=int, default=200)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_query_samples < 1 or args.repeats < 1 or args.warmups < 0:
        raise ValueError("Tham số đo thời gian không hợp lệ.")

    rows = []
    for dataset_name in args.datasets:
        rows.extend(
            benchmark_dataset(
                dataset_name,
                max_query_samples=args.max_query_samples,
                warmups=args.warmups,
                repeats=args.repeats,
            )
        )

    result = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    report_columns = [
        "dataset",
        "config",
        "knn_dimension",
        "median_ms_per_sample",
        "speedup_vs_normal",
    ]
    print(
        result[report_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nĐã lưu: {output_path.resolve()}")


if __name__ == "__main__":
    main()
