import re
import sys
from pathlib import Path

root, n_subset = Path(sys.argv[1]), int(sys.argv[2])  # pipeline output_dir, n_subset_samples


def n_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1


# 1. Generated sets: expected vs actual number of splits
for model_dir in sorted((root / "generated_sequences").iterdir()):
    split_dir = root / "generated_compairr_sequences_split" / model_dir.name
    for gen_file in sorted(model_dir.glob("*.tsv")):
        expected = n_rows(gen_file) // n_subset
        pattern = re.compile(rf"{re.escape(gen_file.stem)}_\d+\.tsv")
        found = sum(1 for f in split_dir.glob("*.tsv") if pattern.fullmatch(f.name))
        if found != expected:
            print(f"SPLITS  {model_dir.name}/{gen_file.stem}: {found}/{expected} splits")

# 2. Train/test reference copies used by the analyses
for split in ["train", "test"]:
    for ref_file in sorted((root / f"{split}_sequences").glob("*.tsv")):
        compairr_file = root / f"{split}_compairr_sequences" / ref_file.name
        if not compairr_file.is_file() or n_rows(compairr_file) != n_rows(ref_file):
            print(f"REF     {split}/{ref_file.name}: compairr copy missing or truncated")

# 3. Experiments where immuneML never produced a copied generated set
for exp_dir in sorted(root.glob("exp_*")):
    for immuneml_dir in sorted(exp_dir.glob("*/*/immuneml")):
        model, data_file = immuneml_dir.parent.parent.name, immuneml_dir.parent.name
        exp = exp_dir.name.split("_")[1]
        if not (root / "generated_sequences" / model / f"{data_file}_{exp}.tsv").is_file():
            gen = list((immuneml_dir / "gen_model/generated_sequences/model").glob("*.tsv"))
            state = f"immuneML output has {n_rows(gen[0])} rows" if gen else "no immuneML generated file"
            print(f"UNFINISHED {exp_dir.name}/{model}/{data_file}: {state}")
