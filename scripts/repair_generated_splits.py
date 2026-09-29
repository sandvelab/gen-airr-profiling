"""Rebuilds CompAIRR files and splits for generated sets whose split count doesn't match their size.

Earlier parallel training runs could read another experiment's CompAIRR file while it was being rewritten,
leaving too few (or truncated) splits. This script redoes preprocessing and splitting sequentially from the
complete files in generated_sequences/. It doesn't retrain anything.

Usage: python scripts/repair_generated_splits.py <output_dir> <n_subset_samples> [--dry_run]
"""
import argparse
import re
from pathlib import Path

from gen_airr_bm.training.training_orchestrator import TrainingOrchestrator
from gen_airr_bm.utils.compairr_utils import preprocess_file_for_compairr


def count_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f) - 1


def find_split_files(split_dir: Path, stem: str) -> list[Path]:
    pattern = re.compile(rf"{re.escape(stem)}_\d+\.tsv")
    return [f for f in split_dir.glob("*.tsv") if pattern.fullmatch(f.name)] if split_dir.is_dir() else []


def main(output_dir: Path, n_subset_samples: int, dry_run: bool) -> None:
    n_repaired = 0
    for model_dir in sorted((output_dir / "generated_sequences").iterdir()):
        compairr_dir = output_dir / "generated_compairr_sequences" / model_dir.name
        split_dir = output_dir / "generated_compairr_sequences_split" / model_dir.name

        for gen_file in sorted(model_dir.glob("*.tsv")):
            expected = count_rows(gen_file) // n_subset_samples
            split_files = find_split_files(split_dir, gen_file.stem)
            if len(split_files) == expected:
                continue

            print(f"{model_dir.name}/{gen_file.stem}: {len(split_files)}/{expected} splits -> rebuilding")
            n_repaired += 1
            if dry_run:
                continue

            for split_file in split_files:
                split_file.unlink()
            compairr_dir.mkdir(parents=True, exist_ok=True)
            preprocess_file_for_compairr(str(model_dir), str(compairr_dir), gen_file.name)
            TrainingOrchestrator.divide_generated_sequences(str(compairr_dir), gen_file.stem, str(split_dir),
                                                            n_subset_samples)

    print(f"{'Would rebuild' if dry_run else 'Rebuilt'} {n_repaired} generated sets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Pipeline output_dir (contains generated_sequences/).")
    parser.add_argument("n_subset_samples", type=int, help="Sequences per split, e.g. 21500.")
    parser.add_argument("--dry_run", action="store_true", help="Only list the sets that would be rebuilt.")
    args = parser.parse_args()
    main(args.output_dir, args.n_subset_samples, args.dry_run)
