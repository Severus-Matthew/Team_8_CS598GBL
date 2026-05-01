#!/usr/bin/env python3
import argparse
from pathlib import Path

import joblib
import numpy as np


def compute_reconstruction_metrics(original_embeddings, reconstructed_embeddings):
    """
    Compute reconstruction quality between original embeddings and reconstructed embeddings.

    Both should be in the original embedding space, not scaled space.
    """
    diff = original_embeddings - reconstructed_embeddings

    mse = np.mean(diff ** 2)

    fro_error = np.linalg.norm(diff, ord="fro")
    fro_original = np.linalg.norm(original_embeddings, ord="fro")
    relative_fro_error = fro_error / (fro_original + 1e-12)

    # Per-sample cosine similarity between original and reconstructed embeddings.
    numerator = np.sum(original_embeddings * reconstructed_embeddings, axis=1)
    denominator = (
        np.linalg.norm(original_embeddings, axis=1)
        * np.linalg.norm(reconstructed_embeddings, axis=1)
        + 1e-12
    )
    cosine_similarities = numerator / denominator
    mean_cosine_similarity = np.mean(cosine_similarities)

    return {
        "reconstruction_mse": mse,
        "relative_frobenius_error": relative_fro_error,
        "mean_reconstruction_cosine_similarity": mean_cosine_similarity,
        "per_sample_reconstruction_cosine_similarity": cosine_similarities,
    }

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
    save_reconstruction=False,
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
        num_negative = int(np.sum(X < 0))
        num_above_one = int(np.sum(X > 1))

        if num_negative > 0 or num_above_one > 0:
            print(
                f"NMF clipping warning: found {num_negative} values < 0 and "
                f"{num_above_one} values > 1 after train-fitted scaling. "
                "Clipping to [0, 1] before NMF transform."
            )

        X = np.clip(X, 0.0, 1.0)

    features = extractor.transform(X)

    print(f"Output feature shape: {features.shape}")

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

    if method == "nmf":
        save_dict["nmf_clipped_to_0_1"] = np.array(True)
        save_dict["nmf_num_values_below_0_before_clipping"] = np.array(num_negative)
        save_dict["nmf_num_values_above_1_before_clipping"] = np.array(num_above_one)

    # Reconstruction evaluation.
    if hasattr(extractor, "inverse_transform"):
        X_reconstructed = extractor.inverse_transform(features)

        if scaler is not None:
            embeddings_reconstructed = scaler.inverse_transform(X_reconstructed)
        else:
            embeddings_reconstructed = X_reconstructed

        metrics = compute_reconstruction_metrics(
            original_embeddings=embeddings,
            reconstructed_embeddings=embeddings_reconstructed,
        )

        for key, value in metrics.items():
            save_dict[key] = value

        print("Reconstruction MSE:", float(metrics["reconstruction_mse"]))
        print(
            "Relative Frobenius error:",
            float(metrics["relative_frobenius_error"]),
        )
        print(
            "Mean reconstruction cosine similarity:",
            float(metrics["mean_reconstruction_cosine_similarity"]),
        )

        if save_reconstruction:
            save_dict["reconstructed_embeddings"] = embeddings_reconstructed
    else:
        print(f"Warning: {method} extractor has no inverse_transform; skipping reconstruction.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{fitted_embedding_key}_{method}_{n_components}"
    output_path = output_dir / f"{prefix}_{split_name}_features.npz"

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

    parser.add_argument(
        "--save-reconstruction",
        action="store_true",
        help="Save reconstructed embeddings in the output .npz file.",
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
            save_reconstruction=args.save_reconstruction,
        )


if __name__ == "__main__":
    main()