#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def read_fasta_files(input_path):
    """
    Read one .fasta/.fa/.faa file or all FASTA files inside a directory.
    Returns list of (record_id, sequence).
    """
    input_path = Path(input_path)

    if input_path.is_file():
        fasta_files = [input_path]
    else:
        fasta_files = []
        for ext in ["*.fasta", "*.fa", "*.faa"]:
            fasta_files.extend(input_path.glob(ext))

    records = []
    for fasta in fasta_files:
        for record in SeqIO.parse(str(fasta), "fasta"):
            seq = str(record.seq).replace(" ", "").replace("\n", "").upper()
            records.append((record.id, seq))

    return records


def pool_single_sequence(residue_emb, cls_emb, pooling):
    """
    Pool one protein sequence into one embedding.

    residue_emb: [num_residues, hidden_dim]
    cls_emb: [hidden_dim]
    """
    if pooling == "cls":
        return cls_emb

    if residue_emb.size(0) == 0:
        raise ValueError("Found an empty residue sequence after removing special tokens.")

    if pooling == "mean":
        return residue_emb.mean(dim=0)
    elif pooling == "min":
        return residue_emb.min(dim=0).values
    elif pooling == "max":
        return residue_emb.max(dim=0).values
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")


def pool_embeddings_multi(last_hidden_state, attention_mask, pooling_methods, hidden_states=None):
    """
    Compute multiple pooled embeddings in one forward pass.

    last_hidden_state: [batch, seq_len, hidden_dim]
    attention_mask: [batch, seq_len]
    hidden_states: optional tuple of layer hidden states.
        Each element has shape [batch, seq_len, hidden_dim].
        Used for layer_mean.

    pooling_methods: list[str], e.g. ["mean", "max", "cls", "layer_mean"]

    Returns:
        dict[str, Tensor], each Tensor has shape [batch, hidden_dim]
    """
    pooled = {method: [] for method in pooling_methods}

    for i in range(last_hidden_state.size(0)):
        valid_len = int(attention_mask[i].sum().item())

        cls_emb = last_hidden_state[i, 0, :]
        residue_emb = last_hidden_state[i, 1:valid_len - 1, :]

        for method in pooling_methods:
            if method == "layer_mean":
                if hidden_states is None:
                    raise ValueError("hidden_states is required for layer_mean pooling.")

                # For each layer:
                #   1. remove CLS/BOS and EOS
                #   2. mean-pool over residues
                # Then average those vectors over all layers.
                layer_mean_embs = []

                for layer_hidden in hidden_states[1:]:
                    layer_residue_emb = layer_hidden[i, 1:valid_len - 1, :]

                    if layer_residue_emb.size(0) == 0:
                        raise ValueError(
                            "Found an empty residue sequence after removing special tokens."
                        )

                    layer_mean_embs.append(layer_residue_emb.mean(dim=0))

                emb = torch.stack(layer_mean_embs, dim=0).mean(dim=0)

            else:
                emb = pool_single_sequence(
                    residue_emb=residue_emb,
                    cls_emb=cls_emb,
                    pooling=method,
                )

            pooled[method].append(emb)

    pooled = {
        method: torch.stack(values, dim=0)
        for method, values in pooled.items()
    }

    return pooled


@torch.no_grad()
def embed_sequences(
    records,
    model_name,
    output_file,
    batch_size=8,
    device=None,
    pooling_methods=None,
    save_per_residue=False,
):
    if pooling_methods is None:
        pooling_methods = ["mean"]

    # Remove duplicates while preserving order.
    pooling_methods = list(dict.fromkeys(pooling_methods))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    print(f"Pooling methods: {pooling_methods}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_ids = []
    all_embeddings = {method: [] for method in pooling_methods}
    per_residue_embeddings = {} if save_per_residue else None

    for start in tqdm(range(0, len(records), batch_size), desc="Embedding"):
        batch = records[start:start + batch_size]
        ids = [x[0] for x in batch]
        seqs = [x[1] for x in batch]

        inputs = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        need_hidden_states = "layer_mean" in pooling_methods

        outputs = model(
            **inputs,
            output_hidden_states=need_hidden_states,
        )

        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        pooled_batch = pool_embeddings_multi(
            last_hidden_state=last_hidden,
            attention_mask=attention_mask,
            pooling_methods=pooling_methods,
            hidden_states=outputs.hidden_states if need_hidden_states else None,
        )

        all_ids.extend(ids)

        for method in pooling_methods:
            all_embeddings[method].append(pooled_batch[method].cpu().numpy())

        if save_per_residue:
            for i, seq_id in enumerate(ids):
                valid_len = int(attention_mask[i].sum().item())
                residue_emb = last_hidden[i, 1:valid_len - 1, :].cpu().numpy()
                per_residue_embeddings[seq_id] = residue_emb

    save_dict = {
        "ids": np.array(all_ids, dtype=object),
        "pooling_methods": np.array(pooling_methods, dtype=object),
    }

    for method in pooling_methods:
        embeddings = np.concatenate(all_embeddings[method], axis=0)
        save_dict[f"embeddings_{method}"] = embeddings
        print(f"{method} embedding shape: {embeddings.shape}")

    if save_per_residue:
        save_dict["per_residue_embeddings"] = per_residue_embeddings

    np.savez_compressed(output_file, **save_dict)

    print(f"Saved embeddings to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract protein sequence embeddings using ESM2-8M."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a FASTA file or a directory containing .fasta/.fa/.faa files.",
    )

    parser.add_argument(
        "--output",
        default="esm2_8m_embeddings.npz",
        help="Output .npz file.",
    )

    parser.add_argument(
        "--model",
        default="facebook/esm2_t6_8M_UR50D",
        help="HuggingFace model name.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for embedding extraction.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device, e.g. cuda, cpu, cuda:0. If not set, auto-detect.",
    )

    parser.add_argument(
        "--pooling",
        nargs="+",
        choices=["mean", "min", "max", "cls", "layer_mean"],
        default=["mean"],
        help=(
            "One or more pooling methods for sequence-level embeddings. "
            "Example: --pooling mean max min cls layer_mean"
        ),
    )


    parser.add_argument(
        "--save-per-residue",
        action="store_true",
        help="Also save per-residue embeddings for each protein.",
    )

    args = parser.parse_args()

    records = read_fasta_files(args.input)

    if len(records) == 0:
        raise ValueError(f"No FASTA records found in {args.input}")

    print(f"Found {len(records)} protein sequences.")

    embed_sequences(
        records=records,
        model_name=args.model,
        output_file=args.output,
        batch_size=args.batch_size,
        device=args.device,
        pooling_methods=args.pooling,
        save_per_residue=args.save_per_residue,
    )


if __name__ == "__main__":
    main()