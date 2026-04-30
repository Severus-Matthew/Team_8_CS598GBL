#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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


def get_scaler(method, scale_mode):
    """
    Return a scaler object or None.

    PCA/ICA usually use StandardScaler.
    NMF requires non-negative input, so MinMaxScaler is safe.
    """
    if scale_mode == "none":
        return None

    if scale_mode == "standard":
        return StandardScaler()

    if scale_mode == "minmax":
        return MinMaxScaler()

    if scale_mode == "auto":
        if method in ["pca", "ica"]:
            return StandardScaler()
        if method == "nmf":
            return MinMaxScaler()

    raise ValueError(f"Unknown scale_mode: {scale_mode}")


def get_extractor(method, n_components, random_state, max_iter):
    if method == "pca":
        return PCA(
            n_components=n_components,
            random_state=random_state,
        )

    if method == "ica":
        return FastICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            whiten="unit-variance",
        )

    if method == "nmf":
        return NMF(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            init="nndsvda",
        )

    raise ValueError(f"Unknown method: {method}")


def fit_one_method(
    ids,
    embeddings,
    method,
    n_components,
    embedding_key,
    output_dir,
    scale_mode,
    random_state,
    max_iter,
):
    print("=" * 80)
    print(f"Fitting method: {method}")
    print(f"Embedding key: {embedding_key}")
    print(f"Input shape: {embeddings.shape}")
    print(f"n_components: {n_components}")
    print(f"scale_mode: {scale_mode}")

    scaler = get_scaler(method, scale_mode)

    if scaler is not None:
        X = scaler.fit_transform(embeddings)
    else:
        X = embeddings

    if method == "nmf":
        X = np.clip(X, a_min=0.0, a_max=None)

    extractor = get_extractor(
        method=method,
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
    )

    features = extractor.fit_transform(X)

    print(f"Feature shape: {features.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{embedding_key}_{method}_{n_components}"

    feature_path = output_dir / f"{prefix}_train_features.npz"
    model_path = output_dir / f"{prefix}_fitted.joblib"

    save_dict = {
        "ids": ids,
        "features": features,
        "method": np.array(method, dtype=object),
        "embedding_key": np.array(embedding_key, dtype=object),
        "n_components": np.array(n_components),
        "scale_mode": np.array(scale_mode, dtype=object),
        "split": np.array("train", dtype=object),
    }

    if method == "pca":
        save_dict["explained_variance_ratio"] = extractor.explained_variance_ratio_
        save_dict["explained_variance"] = extractor.explained_variance_
        print(
            "PCA explained variance ratio sum:",
            float(extractor.explained_variance_ratio_.sum()),
        )

    if method == "ica":
        save_dict["mixing_matrix"] = extractor.mixing_

    if method == "nmf":
        save_dict["components"] = extractor.components_
        save_dict["reconstruction_error"] = np.array(extractor.reconstruction_err_)
        print("NMF reconstruction error:", float(extractor.reconstruction_err_))

    np.savez_compressed(feature_path, **save_dict)

    joblib.dump(
        {
            "scaler": scaler,
            "extractor": extractor,
            "method": method,
            "embedding_key": embedding_key,
            "n_components": n_components,
            "scale_mode": scale_mode,
            "random_state": random_state,
            "max_iter": max_iter,
        },
        model_path,
    )

    print(f"Saved train features to: {feature_path}")
    print(f"Saved fitted model to:   {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit PCA / ICA / NMF feature extractors on training embeddings only."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Training .npz embedding file.",
    )

    parser.add_argument(
        "--embedding-key",
        default="embeddings_mean",
        help=(
            "Embedding key in the .npz file. "
            "Examples: embeddings_mean, embeddings_max, embeddings_min, "
            "embeddings_cls, embeddings_layer_mean."
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["pca", "ica", "nmf"],
        default=["pca"],
        help="Feature extraction methods to fit.",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of components.",
    )

    parser.add_argument(
        "--output-dir",
        default="fitted_feature_extractors",
        help="Directory to save fitted models and train features.",
    )

    parser.add_argument(
        "--scale-mode",
        choices=["auto", "standard", "minmax", "none"],
        default="auto",
        help=(
            "Scaling mode. "
            "auto = StandardScaler for PCA/ICA, MinMaxScaler for NMF."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for ICA/NMF.",
    )

    args = parser.parse_args()

    ids, embeddings = load_embeddings(args.input, args.embedding_key)

    for method in args.methods:
        fit_one_method(
            ids=ids,
            embeddings=embeddings,
            method=method,
            n_components=args.n_components,
            embedding_key=args.embedding_key,
            output_dir=args.output_dir,
            scale_mode=args.scale_mode,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )


if __name__ == "__main__":
    main()