"""Splits regenerated TCRsep sequences into the training-time generated set and the resampled set.

Older immuneML versions seeded every TCRsep generation with the same seed, so models trained on different
repertoires generated largely the same sequences, both at the end of training and when resampling. The trained
models are fine, so both sets are regenerated in one resampling run of n_generated + n_resampled sequences. This
script moves the first n_generated rows of each file in resampled_sequences_raw/ to generated_sequences/, keeps the
rest as the resampled set, and rebuilds the CompAIRR files and splits of the generated set. It doesn't retrain
anything. The old outputs have to be moved away first; files that already have a generated set are skipped, so
the script can be rerun.

Usage: python scripts/split_regenerated_tcrsep.py <output_dir> <n_generated> <n_subset_samples> [--dry_run]
"""
import argparse
from pathlib import Path

import pandas as pd

from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator
from gen_airr_bm.utils.compairr_utils import preprocess_file_for_compairr

MODELS = ["TCRsep_overlap", "TCRsep_kmer"]


def main(output_dir: Path, n_generated: int, n_subset_samples: int, dry_run: bool) -> None:
    n_split = 0
    for model in MODELS:
        resampled_dir = output_dir / "resampled_sequences_raw" / model
        generated_dir = output_dir / "generated_sequences" / model
        compairr_dir = output_dir / "generated_compairr_sequences" / model
        split_dir = output_dir / "generated_compairr_sequences_split" / model

        for resampled_file in sorted(resampled_dir.glob("*.tsv")):
            generated_file = generated_dir / resampled_file.name
            if generated_file.exists():
                print(f"{model}/{resampled_file.stem}: generated set already exists -> skipping")
                continue

            sequences = pd.read_csv(resampled_file, sep="\t")
            if len(sequences) <= n_generated:
                raise ValueError(f"{resampled_file} has {len(sequences)} rows, but more than {n_generated} are "
                                 f"needed. Was it already split, or resampled with the old n_samples?")

            print(f"{model}/{resampled_file.stem}: {n_generated} generated, {len(sequences) - n_generated} resampled")
            n_split += 1
            if dry_run:
                continue

            generated_dir.mkdir(parents=True, exist_ok=True)
            sequences.iloc[:n_generated].to_csv(generated_file, sep="\t", index=False)
            sequences.iloc[n_generated:].to_csv(resampled_file, sep="\t", index=False)

            for split_file in TrainingOrchestrator.get_split_files(split_dir, generated_file.stem):
                split_file.unlink()
            compairr_dir.mkdir(parents=True, exist_ok=True)
            preprocess_file_for_compairr(str(generated_dir), str(compairr_dir), generated_file.name)
            TrainingOrchestrator.divide_generated_sequences(str(compairr_dir), generated_file.stem, str(split_dir),
                                                            n_subset_samples)

    print(f"{'Would split' if dry_run else 'Split'} {n_split} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Pipeline output_dir (contains resampled_sequences_raw/).")
    parser.add_argument("n_generated", type=int, help="Sequences generated at the end of training, e.g. 130000.")
    parser.add_argument("n_subset_samples", type=int, help="Sequences per split, e.g. 21500.")
    parser.add_argument("--dry_run", action="store_true", help="Only list the files that would be split.")
    args = parser.parse_args()
    main(args.output_dir, args.n_generated, args.n_subset_samples, args.dry_run)
