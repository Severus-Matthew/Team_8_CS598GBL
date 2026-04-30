#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np


def load_embeddings(npz_path, embedding_key):
    data = np.load(npz_path, allow_pickle=True)

    if "ids" not in data:
        raise KeyError(f"'ids' not found in {npz_path}")

    if embedding_key not in data:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    ids = data["ids"]
    embeddings = data[embedding_key]

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    return ids, embeddings


def discover_fitted_models(fitted_dir, methods=None):
    fitted_dir = Path(fitted_dir)

    if not fitted_dir.exists():
        raise FileNotFoundError(f"Fitted directory does not exist: {fitted_dir}")

    model_paths = sorted(fitted_dir.glob("*_fitted.joblib"))

    if methods is not None:
        methods = set(methods)
        filtered = []

        for path in model_paths:
            obj = joblib.load(path)
            if obj["method"] in methods:
                filtered.append(path)

        model_paths = filtered

    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No fitted models found in {fitted_dir}. "
            f"Expected files ending with '_fitted.joblib'."
        )

    return model_paths


def apply_one_model(
    ids,
    embeddings,
    model_path,
    input_embedding_key,
    output_dir,
    split_name,
):
    obj = joblib.load(model_path)

    scaler = obj["scaler"]
    extractor = obj["extractor"]
    method = obj["method"]
    fitted_embedding_key = obj["embedding_key"]
    n_components = obj["n_components"]
    scale_mode = obj["scale_mode"]

    if input_embedding_key != fitted_embedding_key:
        raise ValueError(
            f"Input embedding key '{input_embedding_key}' does not match "
            f"fitted model embedding key '{fitted_embedding_key}' from {model_path}."
        )

    print("=" * 80)
    print(f"Applying fitted model: {model_path}")
    print(f"Method: {method}")
    print(f"Embedding key: {fitted_embedding_key}")
    print(f"Input shape: {embeddings.shape}")
    print(f"n_components: {n_components}")
    print(f"scale_mode: {scale_mode}")

    if scaler is not None:
        X = scaler.transform(embeddings)
    else:
        X = embeddings

    if method == "nmf":
        # Val/test values can fall outside the train min/max range,
        # so MinMaxScaler may produce small negative values.
        # NMF requires non-negative inputs, so clip safely.
        X = np.clip(X, a_min=0.0, a_max=None)

    features = extractor.transform(X)

    print(f"Output feature shape: {features.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{fitted_embedding_key}_{method}_{n_components}"
    output_path = output_dir / f"{prefix}_{split_name}_features.npz"

    save_dict = {
        "ids": ids,
        "features": features,
        "method": np.array(method, dtype=object),
        "embedding_key": np.array(fitted_embedding_key, dtype=object),
        "n_components": np.array(n_components),
        "scale_mode": np.array(scale_mode, dtype=object),
        "split": np.array(split_name, dtype=object),
        "fitted_model_path": np.array(str(model_path), dtype=object),
    }

    np.savez_compressed(output_path, **save_dict)

    print(f"Saved transformed features to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply fitted PCA / ICA / NMF feature extractors to val/test/inference embeddings."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input .npz embedding file for val/test/inference.",
    )

    parser.add_argument(
        "--embedding-key",
        default="embeddings_mean",
        help=(
            "Embedding key in the .npz file. Must match the key used during fitting."
        ),
    )

    parser.add_argument(
        "--fitted-dir",
        required=True,
        help="Directory containing *_fitted.joblib files from fit_feature_extractors.py.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pca", "ica", "nmf"],
        default=None,
        help=(
            "Optional subset of methods to apply. "
            "If omitted, apply all fitted models found in --fitted-dir."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="transformed_features",
        help="Directory to save transformed features.",
    )

    parser.add_argument(
        "--split-name",
        default="eval",
        help="Name to include in output file, e.g. val, test, inference.",
    )

    args = parser.parse_args()

    ids, embeddings = load_embeddings(args.input, args.embedding_key)

    model_paths = discover_fitted_models(
        fitted_dir=args.fitted_dir,
        methods=args.methods,
    )

    for model_path in model_paths:
        apply_one_model(
            ids=ids,
            embeddings=embeddings,
            model_path=model_path,
            input_embedding_key=args.embedding_key,
            output_dir=args.output_dir,
            split_name=args.split_name,
        )


if __name__ == "__main__":
    main()