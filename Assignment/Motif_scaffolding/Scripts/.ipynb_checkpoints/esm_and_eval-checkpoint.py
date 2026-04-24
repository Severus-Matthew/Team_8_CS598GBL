import os
import re
import csv
import glob
import json
import shutil
import subprocess
from pathlib import Path

import torch
import numpy as np
from Bio.PDB import PDBParser, Superimposer
from transformers import EsmForProteinFolding, AutoTokenizer


BACKBONE_BASE = Path("/home/ubuntu/manvi/Team_8_CS598GBL/outputs/custom90_128_motif_scaffolding/default")
MPNN_BASE = Path("/home/ubuntu/manvi/Team_8_CS598GBL/outputs/mpnn_custom90_128_motif_scaffolding/default")
OUT_BASE = Path("/home/ubuntu/manvi/Team_8_CS598GBL/outputs/esmfold_eval_custom90_128_motif_scaffolding/default")

NUM_SEQS_PER_TARGET = 1   # set 5/10 if you want evaluate more MPNN samples per backbone
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean_sequence(seq: str) -> str:
    seq = seq.strip().replace("/", "")
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)
    return seq


def read_mpnn_fasta_sequences(fasta_path):
    records = []
    header = None
    chunks = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, clean_sequence("".join(chunks))))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        records.append((header, clean_sequence("".join(chunks))))

    # ProteinMPNN usually writes original/native first, then sampled sequences.
    sampled = []
    for h, s in records:
        if "sample=" in h or h.startswith("T="):
            sampled.append((h, s))

    if sampled:
        return sampled

    # fallback: skip first if multiple records
    if len(records) > 1:
        return records[1:]

    return records


def find_backbone_pdb(scaffold_dir: Path):
    pdbs = sorted(scaffold_dir.glob("*.pdb"))
    if pdbs:
        return pdbs[0]

    pdbs = sorted(scaffold_dir.rglob("*.pdb"))
    return pdbs[0] if pdbs else None


def find_mpnn_fasta(mpnn_dir: Path):
    candidates = []
    for ext in ["*.fa", "*.fasta"]:
        candidates.extend(mpnn_dir.rglob(ext))
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def get_ca_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atoms.append(residue["CA"])
        break
    return atoms


def calc_ca_rmsd(ref_pdb, pred_pdb):
    ref_atoms = get_ca_atoms(ref_pdb)
    pred_atoms = get_ca_atoms(pred_pdb)

    n = min(len(ref_atoms), len(pred_atoms))
    if n == 0:
        return None, 0

    ref_atoms = ref_atoms[:n]
    pred_atoms = pred_atoms[:n]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, pred_atoms)
    return float(sup.rms), n


def run_usalign(pred_pdb, ref_pdb):
    usalign = shutil.which("USalign") or shutil.which("./USalign")
    if usalign is None:
        local = Path("./USalign")
        if local.exists():
            usalign = str(local.resolve())

    if usalign is None:
        return None, None

    try:
        result = subprocess.run(
            [usalign, str(pred_pdb), str(ref_pdb)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        text = result.stdout + "\n" + result.stderr
        scores = re.findall(r"TM-score=\s*([0-9.]+)", text)
        scores = [float(x) for x in scores]
        if len(scores) >= 2:
            return scores[0], scores[1]
        if len(scores) == 1:
            return scores[0], None
        return None, None
    except Exception:
        return None, None


def fold_sequence(model, tokenizer, seq, out_pdb):
    inputs = tokenizer([seq], return_tensors="pt", add_special_tokens=False).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    pdb = model.output_to_pdb(outputs)[0]

    with open(out_pdb, "w") as f:
        f.write(pdb)

    mean_plddt = None
    if hasattr(outputs, "plddt"):
        mean_plddt = float(outputs.plddt[0].detach().cpu().mean().item())
    elif "plddt" in outputs:
        mean_plddt = float(outputs["plddt"][0].detach().cpu().mean().item())

    return mean_plddt


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    print("Loading ESMFold...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model.eval()
    model.trunk.set_chunk_size(64)

    results_by_type = {}

    len_dirs = sorted(BACKBONE_BASE.glob("len_*"))

    for len_dir in len_dirs:
        if not len_dir.is_dir():
            continue

        length_name = len_dir.name
        length_match = re.search(r"len_(\d+)", length_name)
        protein_length = int(length_match.group(1)) if length_match else None

        for scaffold_dir in sorted(len_dir.iterdir()):
            if not scaffold_dir.is_dir():
                continue

            scaffold_type = scaffold_dir.name
            rel = scaffold_dir.relative_to(BACKBONE_BASE)

            backbone_pdb = find_backbone_pdb(scaffold_dir)
            mpnn_dir = MPNN_BASE / rel
            fasta_path = find_mpnn_fasta(mpnn_dir)

            if backbone_pdb is None:
                print(f"[SKIP] No backbone PDB: {scaffold_dir}")
                continue

            if fasta_path is None:
                print(f"[SKIP] No MPNN fasta: {mpnn_dir}")
                continue

            seq_records = read_mpnn_fasta_sequences(fasta_path)
            if not seq_records:
                print(f"[SKIP] No sequences found: {fasta_path}")
                continue

            seq_records = seq_records[:NUM_SEQS_PER_TARGET]

            for sample_idx, (header, seq) in enumerate(seq_records):
                if not seq:
                    print(f"[SKIP] Empty seq: {fasta_path}")
                    continue

                out_dir = OUT_BASE / rel
                out_dir.mkdir(parents=True, exist_ok=True)

                pred_pdb = out_dir / f"esmfold_sample_{sample_idx}.pdb"

                print(f"[RUN] {rel} sample={sample_idx} len={len(seq)}")

                try:
                    mean_plddt = fold_sequence(model, tokenizer, seq, pred_pdb)
                    ca_rmsd, n_ca = calc_ca_rmsd(backbone_pdb, pred_pdb)
                    tm1, tm2 = run_usalign(pred_pdb, backbone_pdb)

                    row = {
                        "length_dir": length_name,
                        "protein_length_from_dir": protein_length,
                        "scaffold_type": scaffold_type,
                        "sample_idx": sample_idx,
                        "sequence_length": len(seq),
                        "backbone_pdb": str(backbone_pdb),
                        "mpnn_fasta": str(fasta_path),
                        "esmfold_pdb": str(pred_pdb),
                        "mpnn_header": header,
                        "ca_rmsd": ca_rmsd,
                        "num_ca_aligned": n_ca,
                        "mean_plddt": mean_plddt,
                        "tm_score_1": tm1,
                        "tm_score_2": tm2,
                    }

                    results_by_type.setdefault(scaffold_type, []).append(row)

                except Exception as e:
                    print(f"[ERROR] {rel} sample={sample_idx}: {e}")

    csv_dir = OUT_BASE / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for scaffold_type, rows in results_by_type.items():
        all_rows.extend(rows)
        csv_path = csv_dir / f"{scaffold_type}_esmfold_eval.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"[SAVED] {csv_path}")

    if all_rows:
        all_csv = csv_dir / "all_scaffold_types_esmfold_eval.csv"
        with open(all_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"[SAVED] {all_csv}")


if __name__ == "__main__":
    main()