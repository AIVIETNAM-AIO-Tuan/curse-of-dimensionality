"""Minimal PCA/LDA/KNN ablation study for Google Colab.

The script produces numerical tables only. It evaluates four configurations
on identical stratified folds:

    normal, pca, lda, pca_lda

Outputs:
    cv_fold_results.csv
    cv_summary.csv
    cv_delta_vs_normal.csv
    query_time.csv

Important scope:
    - PCA component counts are kept at the values used in the original report
      so this run isolates the effect of stratified CV and fixed brute-force
      KNN. It does not solve PCA component selection.
    - Query time measures only KNN.predict, not PCA/LDA transformation time.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_lfw_people, fetch_openml, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
N_NEIGHBORS = 5
QUERY_TIMING_MAX_SAMPLES = 200
QUERY_TIMING_REPEATS = 5

CONFIG_ORDER = ["normal", "pca", "lda", "pca_lda"]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

# Kept unchanged from the original report for a controlled comparison.
PCA_COMPONENTS = {
    "breast_cancer": 8,
    "lfw": 100,
    "mnist": 57,
}

LDA_COMPONENTS = {
    "breast_cancer": 1,
    "lfw": 61,
    "mnist": 9,
}

DISPLAY_NAMES = {
    "breast_cancer": "Breast Cancer",
    "lfw": "LFW People",
    "mnist": "MNIST",
}


def load_dataset(dataset_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one dataset and return dense float32 features and integer labels."""

    if dataset_key == "breast_cancer":
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
    elif dataset_key == "lfw":
        dataset = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
        X, y = dataset.data, dataset.target
    elif dataset_key == "mnist":
        dataset = fetch_openml(
            "mnist_784",
            version=1,
            parser="auto",
            as_frame=False,
        )
        X, y = dataset.data, dataset.target.astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, y


def safe_pca_components(requested: int, X_train: np.ndarray) -> int:
    """Clip PCA components to the rank limit of the current training fold."""

    return min(requested, X_train.shape[0] - 1, X_train.shape[1])


def safe_lda_components(
    requested: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> int:
    """Clip LDA components to min(input dimension, number of classes - 1)."""

    return min(requested, X_train.shape[1], np.unique(y_train).size - 1)


def build_representations(
    dataset_key: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    fold_seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit every preprocessing step on the training fold only."""

    scaler = StandardScaler()
    X_train_scaled = np.ascontiguousarray(
        scaler.fit_transform(X_train), dtype=np.float32
    )
    X_test_scaled = np.ascontiguousarray(
        scaler.transform(X_test), dtype=np.float32
    )

    pca_components = safe_pca_components(
        PCA_COMPONENTS[dataset_key], X_train_scaled
    )
    pca = PCA(
        n_components=pca_components,
        svd_solver="randomized",
        random_state=fold_seed,
    )
    X_train_pca = np.ascontiguousarray(
        pca.fit_transform(X_train_scaled), dtype=np.float32
    )
    X_test_pca = np.ascontiguousarray(
        pca.transform(X_test_scaled), dtype=np.float32
    )

    lda_components = safe_lda_components(
        LDA_COMPONENTS[dataset_key], X_train_scaled, y_train
    )
    lda = LinearDiscriminantAnalysis(
        n_components=lda_components,
        solver="svd",
    )
    X_train_lda = np.ascontiguousarray(
        lda.fit_transform(X_train_scaled, y_train), dtype=np.float32
    )
    X_test_lda = np.ascontiguousarray(
        lda.transform(X_test_scaled), dtype=np.float32
    )

    pca_lda_components = safe_lda_components(
        LDA_COMPONENTS[dataset_key], X_train_pca, y_train
    )
    pca_lda = LinearDiscriminantAnalysis(
        n_components=pca_lda_components,
        solver="svd",
    )
    X_train_pca_lda = np.ascontiguousarray(
        pca_lda.fit_transform(X_train_pca, y_train), dtype=np.float32
    )
    X_test_pca_lda = np.ascontiguousarray(
        pca_lda.transform(X_test_pca), dtype=np.float32
    )

    return {
        "normal": (X_train_scaled, X_test_scaled),
        "pca": (X_train_pca, X_test_pca),
        "lda": (X_train_lda, X_test_lda),
        "pca_lda": (X_train_pca_lda, X_test_pca_lda),
    }


def make_knn() -> KNeighborsClassifier:
    """Return the same exact brute-force KNN for every configuration."""

    return KNeighborsClassifier(
        n_neighbors=N_NEIGHBORS,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the four report metrics."""

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def median_query_time_ms_per_sample(
    knn: KNeighborsClassifier,
    X_test: np.ndarray,
) -> tuple[float, int]:
    """Measure median KNN.predict time on a bounded query batch."""

    query_sample_count = min(QUERY_TIMING_MAX_SAMPLES, len(X_test))
    X_query = X_test[:query_sample_count]

    knn.predict(X_query[: min(32, query_sample_count)])

    measurements = []
    for _ in range(QUERY_TIMING_REPEATS):
        start = perf_counter()
        knn.predict(X_query)
        measurements.append(perf_counter() - start)

    median_ms_per_sample = (
        1000.0 * float(np.median(measurements)) / query_sample_count
    )
    return median_ms_per_sample, query_sample_count


def evaluate_dataset(
    dataset_key: str,
    n_splits: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all four configurations on identical repeated stratified folds."""

    display_name = DISPLAY_NAMES[dataset_key]
    print(f"\n{'=' * 78}\n{display_name}: loading data")
    X, y = load_dataset(dataset_key)
    print(
        f"shape={X.shape}, classes={np.unique(y).size}, "
        f"CV={n_splits} folds x {n_repeats} repeats"
    )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    fold_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    dataset_start = perf_counter()

    for fold_index, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        fold_start = perf_counter()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        representations = build_representations(
            dataset_key=dataset_key,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            fold_seed=RANDOM_STATE + fold_index,
        )

        for config in CONFIG_ORDER:
            X_train_config, X_test_config = representations[config]
            knn = make_knn()
            knn.fit(X_train_config, y_train)
            y_pred = knn.predict(X_test_config)

            row = {
                "dataset": display_name,
                "fold": fold_index,
                "config": config,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "input_dim": X.shape[1],
                "knn_dim": X_train_config.shape[1],
            }
            row.update(classification_metrics(y_test, y_pred))
            fold_rows.append(row)

            # Query time is descriptive and measured on the first fold only.
            if fold_index == 1:
                query_time, query_count = median_query_time_ms_per_sample(
                    knn, X_test_config
                )
                query_rows.append(
                    {
                        "dataset": display_name,
                        "config": config,
                        "knn_dim": X_train_config.shape[1],
                        "query_samples": query_count,
                        "query_time_ms_per_sample": query_time,
                    }
                )

        fold_seconds = perf_counter() - fold_start
        current = pd.DataFrame(fold_rows)
        current = current[
            (current["dataset"] == display_name)
            & (current["fold"] == fold_index)
        ][["config", "accuracy", "f1_macro"]]
        print(f"fold {fold_index:02d} | {fold_seconds:8.1f} s")
        print(current.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

        del representations, X_train, X_test, y_train, y_test
        gc.collect()

    elapsed = perf_counter() - dataset_start
    print(f"{display_name} completed in {elapsed / 60:.1f} minutes")
    return pd.DataFrame(fold_rows), pd.DataFrame(query_rows)


def summarize_cv(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation across CV folds."""

    summary = (
        fold_results.groupby(["dataset", "config"], sort=False)[METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def paired_delta_vs_normal(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired fold-wise metric differences against normal KNN."""

    delta_rows: list[dict[str, object]] = []

    for (dataset, fold), group in fold_results.groupby(["dataset", "fold"]):
        normal = group[group["config"] == "normal"].iloc[0]
        for config in CONFIG_ORDER[1:]:
            current = group[group["config"] == config].iloc[0]
            row: dict[str, object] = {
                "dataset": dataset,
                "fold": fold,
                "config": config,
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = current[metric] - normal[metric]
            delta_rows.append(row)

    deltas = pd.DataFrame(delta_rows)
    delta_metrics = [f"delta_{metric}" for metric in METRICS]
    summary = (
        deltas.groupby(["dataset", "config"], sort=False)[delta_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(column).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    std_columns = [column for column in summary if column.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0.0)
    return summary


def add_query_speedup(query_results: pd.DataFrame) -> pd.DataFrame:
    """Add normal-query-time / current-query-time within each dataset."""

    output = query_results.copy()
    output["speedup_vs_normal"] = np.nan

    for dataset, indices in output.groupby("dataset").groups.items():
        group = output.loc[indices]
        normal_time = group.loc[
            group["config"] == "normal", "query_time_ms_per_sample"
        ].iloc[0]
        output.loc[indices, "speedup_vs_normal"] = (
            normal_time / group["query_time_ms_per_sample"]
        )

    return output


def save_outputs(
    fold_results: pd.DataFrame,
    query_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all report-ready numerical outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cv_summary = summarize_cv(fold_results)
    delta_summary = paired_delta_vs_normal(fold_results)
    query_summary = add_query_speedup(query_results)

    fold_results.to_csv(output_dir / "cv_fold_results.csv", index=False)
    cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)
    delta_summary.to_csv(output_dir / "cv_delta_vs_normal.csv", index=False)
    query_summary.to_csv(output_dir / "query_time.csv", index=False)

    print("\nCV SUMMARY: mean ± std")
    display_columns = [
        "dataset",
        "config",
        "accuracy_mean",
        "accuracy_std",
        "f1_macro_mean",
        "f1_macro_std",
    ]
    print(
        cv_summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nPAIRED DELTA VS NORMAL: positive means improvement")
    delta_columns = [
        "dataset",
        "config",
        "delta_accuracy_mean",
        "delta_accuracy_std",
        "delta_f1_macro_mean",
        "delta_f1_macro_std",
    ]
    print(
        delta_summary[delta_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:+.4f}",
        )
    )

    print("\nQUERY TIME: first stratified fold only")
    print(
        query_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Results-only PCA/LDA/KNN ablation for Colab"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "final"],
        default="quick",
        help=(
            "quick: 3-fold stratified CV; final: 3-fold x 2-repeat "
            "stratified CV"
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["breast_cancer", "lfw", "mnist"],
        default=["breast_cancer", "lfw", "mnist"],
    )
    parser.add_argument(
        "--output-dir",
        default="cv_results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_splits = 3
    n_repeats = 1 if args.mode == "quick" else 2
    output_dir = Path(args.output_dir)

    print("ENVIRONMENT")
    print(f"Python={platform.python_version()}")
    print(f"NumPy={np.__version__}")
    print(f"scikit-learn={sklearn.__version__}")
    print(f"CPU logical cores={os.cpu_count()}")
    print(f"mode={args.mode}: {n_splits} folds x {n_repeats} repeats")
    print(f"datasets={args.datasets}")

    all_fold_results: list[pd.DataFrame] = []
    all_query_results: list[pd.DataFrame] = []

    for dataset_key in args.datasets:
        fold_results, query_results = evaluate_dataset(
            dataset_key=dataset_key,
            n_splits=n_splits,
            n_repeats=n_repeats,
        )
        all_fold_results.append(fold_results)
        all_query_results.append(query_results)

        # Checkpoint after every dataset so a Colab interruption does not erase
        # previously completed results.
        save_outputs(
            fold_results=pd.concat(all_fold_results, ignore_index=True),
            query_results=pd.concat(all_query_results, ignore_index=True),
            output_dir=output_dir,
        )

    print(f"\nSaved numerical outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
