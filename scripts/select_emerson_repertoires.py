#!/usr/bin/env python3
"""Randomly select Emerson HIP repertoires for tuning and for the experiment, in two separate runs.

Only HIP donors (subject_id P...; the Keck cohort is left out) with a clean_count of at least
--min_clean_count are eligible. Each selected file is copied and renamed to
<subject_id>_<sex>_<CMV status>.tsv:
  Male, CMV True     P00492.tsv -> P00492_male_pos.tsv
  Female, CMV False  P00404.tsv -> P00404_female_neg.tsv

Every selected subject is added to a selection CSV (default <data_dir>/emerson_selection.csv), and
subjects already listed there are never selected again, so the experiment run can't pick the
tuning repertoire.

  tuning      pick --n_select (default 1) repertoires and copy them to <data_dir>/tuning/.
              Can only be run once.
  experiment  pick --n_select (default 100) repertoires and copy them to
              <data_dir>/experimental/emerson_exp1/. Can be run again to extend the experiment,
              e.g. --n_select 20 now and --n_select 80 later for 100 in total. The run column in
              the selection CSV says which run each subject came from.

Example:
    python scripts/select_emerson_repertoires.py tuning data/emerson_updated_metadata.csv path/to/repertoires/ data/
    python scripts/select_emerson_repertoires.py experiment data/emerson_updated_metadata.csv path/to/repertoires/ data/ --n_select 20
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

CMV_SUFFIXES = {"true": "pos", "false": "neg"}
HIP_PREFIX = "P"  # HIP subject_ids are P00001...; Keck ones are Keck0001...
TARGET_DIRS = {"tuning": Path("tuning") / "EMERSON_EXP1", "experiment": Path("experimental") / "emerson_exp1"}
DEFAULT_N_SELECT = {"tuning": 1, "experiment": 100}
SELECTION_COLUMNS = ["role", "run", "subject_id", "CMV", "sex", "age", "clean_count", "filename", "copied_as", "seed"]


def updated_filename(row):
    """Return the filename with sex and CMV status, e.g. P00492.tsv -> P00492_male_pos.tsv."""
    return f'{row["subject_id"].strip()}_{row["sex"].strip().lower()}_{CMV_SUFFIXES[row["CMV"].strip().lower()]}.tsv'


def read_selection(path):
    """Return the rows of the selection CSV, or an empty list if it doesn't exist yet."""
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("role", choices=TARGET_DIRS, help="which set of repertoires to select")
    parser.add_argument("metadata", type=Path, help="metadata CSV file")
    parser.add_argument("repertoire_dir", type=Path, help="folder with the repertoire files")
    parser.add_argument("data_dir", type=Path, help="the project's data folder, e.g. data/")
    parser.add_argument("--selection_file", type=Path, help="default: <data_dir>/emerson_selection.csv")
    parser.add_argument("--min_clean_count", type=int, default=100000)
    parser.add_argument("--n_select", type=int, help="default: 1 for tuning, 100 for experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="print the selection without copying")
    args = parser.parse_args()

    selection_file = args.selection_file or args.data_dir / "emerson_selection.csv"
    target_dir = args.data_dir / TARGET_DIRS[args.role]
    n_select = args.n_select or DEFAULT_N_SELECT[args.role]

    selection = read_selection(selection_file)
    previous_runs = {row["run"] for row in selection if row["role"] == args.role}
    if args.role == "tuning" and previous_runs:
        sys.exit(f"Error: {selection_file} already has tuning repertoires. Remove them first to redo the selection.")
    run = len(previous_runs) + 1
    already_selected = {row["subject_id"] for row in selection}

    with open(args.metadata, newline="") as f:
        rows = list(csv.DictReader(f))

    # Keep HIP rows with enough clean sequences and a known CMV status that weren't selected before
    eligible = [
        row for row in rows
        if row["subject_id"].strip().startswith(HIP_PREFIX)
        and row["subject_id"].strip() not in already_selected
        and row["clean_count"].strip()
        and int(row["clean_count"]) >= args.min_clean_count
        and row["CMV"].strip().lower() in CMV_SUFFIXES
    ]
    print(
        f"{len(eligible)} of {len(rows)} rows are HIP, have clean_count >= {args.min_clean_count} and a CMV "
        f"status, and weren't selected before ({len(already_selected)} excluded)"
    )
    if len(eligible) < n_select:
        sys.exit(f"Error: only {len(eligible)} eligible rows, but {n_select} are needed.")

    selected = random.Random(args.seed).sample(eligible, n_select)

    # Check all source files exist before copying anything
    missing = [row["filename"] for row in selected if not (args.repertoire_dir / row["filename"].strip()).is_file()]
    if missing:
        sys.exit(f"Error: {len(missing)} selected files not found in {args.repertoire_dir}: {', '.join(missing)}")

    n_pos = sum(row["CMV"].strip().lower() == "true" for row in selected)
    print(f"{args.role} run {run}: {len(selected)} files ({n_pos} CMV+, {len(selected) - n_pos} CMV-)")

    if not args.dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        source = args.repertoire_dir / row["filename"].strip()
        target = target_dir / updated_filename(row)
        print(f"{source} -> {target}")
        if not args.dry_run:
            shutil.copy2(source, target)

    if args.dry_run:
        return
    # Record the selection, so later runs don't pick these subjects again
    write_header = not selection_file.is_file()
    selection_file.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTION_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        for row in selected:
            writer.writerow({
                "role": args.role,
                "run": run,
                **{column: row[column].strip() for column in ["subject_id", "CMV", "sex", "age", "clean_count", "filename"]},
                "copied_as": updated_filename(row),
                "seed": args.seed,
            })
    print(f"Added {len(selected)} {args.role} subjects to {selection_file}")


if __name__ == "__main__":
    main()
