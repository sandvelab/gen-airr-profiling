#!/usr/bin/env python3
"""Copy a metadata file to a new folder, adding sequence counts for each repertoire.

Each metadata row's repertoire file is looked up in the repertoire folder by trying:
  1. the file named in the row's filename column
  2. <subject_id>.tsv, e.g. P00492 -> P00492.tsv
  3. the only file named <subject_id>_*.tsv, e.g. Keck0060 -> Keck0060_MC1.tsv

Two columns are added to the copy:
  sequence_count  number of sequences (lines after the header)
  clean_count     number of sequences where cdr3_amino_acid is non-empty and
                  has no "*", and v_resolved and j_resolved are non-empty

Example:
    python add_sequence_counts.py data/metadata.csv data/ metadata_with_counts/
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CLEAN_COLUMNS = ["cdr3_amino_acid", "v_resolved", "j_resolved"]
BUFFER_SIZE = 1024 * 1024  # read repertoire files 1 MB at a time


def find_repertoire_file(row, repertoire_dir, tsv_files):
    """Return the repertoire file for a metadata row, or None if there isn't exactly one."""
    filename = (row.get("filename") or "").strip()
    if filename and (repertoire_dir / filename).is_file():
        return repertoire_dir / filename

    subject_id = (row.get("subject_id") or "").strip()
    if not subject_id:
        return None
    if (repertoire_dir / f"{subject_id}.tsv").is_file():
        return repertoire_dir / f"{subject_id}.tsv"

    matches = [path for path in tsv_files if path.name.startswith(f"{subject_id}_")]
    return matches[0] if len(matches) == 1 else None


def count_sequences(path):
    """Return (sequence_count, clean_count) for one repertoire file.

    clean_count is None if the file doesn't have all of CLEAN_COLUMNS.
    """
    with open(path, "rb", buffering=BUFFER_SIZE) as f:
        header = f.readline().decode("utf-8-sig").rstrip("\r\n").split("\t")
        if not all(column in header for column in CLEAN_COLUMNS):
            return sum(1 for _ in f), None

        # Column positions counted from the end of the line (-1 is the last column).
        # These columns are near the end, so only the end of each line is split,
        # which is about twice as fast as splitting all columns.
        cdr3, v, j = (header.index(column) - len(header) for column in CLEAN_COLUMNS)
        n_splits = -min(cdr3, v, j)

        sequences = clean = 0
        for line in f:
            sequences += 1
            fields = line.rsplit(b"\t", n_splits)
            if (
                len(fields) > n_splits
                and fields[cdr3].strip()
                and b"*" not in fields[cdr3]
                and fields[v].strip()
                and fields[j].strip()
            ):
                clean += 1
    return sequences, clean


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("metadata", type=Path, help="metadata CSV file")
    parser.add_argument("repertoire_dir", type=Path, help="folder with the repertoire files")
    parser.add_argument("output_dir", type=Path, help="folder to write the new metadata file to")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of files to process at once, each in its own process (default: 4)",
    )
    args = parser.parse_args()

    if not args.repertoire_dir.is_dir():
        sys.exit(f"Error: repertoire folder {args.repertoire_dir} does not exist.")
    output_path = args.output_dir / args.metadata.name
    if output_path.resolve() == args.metadata.resolve():
        sys.exit("Error: output folder must differ from the metadata file's folder.")

    with open(args.metadata, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "subject_id" not in fieldnames and "filename" not in fieldnames:
        sys.exit(f"Error: {args.metadata} needs a subject_id or filename column.")

    # Find each row's repertoire file
    tsv_files = sorted(args.repertoire_dir.glob("*.tsv"))
    row_paths = []
    for row in rows:
        path = find_repertoire_file(row, args.repertoire_dir, tsv_files)
        row_paths.append(path)
        if path is None:
            subject_id = (row.get("subject_id") or "").strip() or "(empty)"
            filename = (row.get("filename") or "").strip() or "(empty)"
            print(
                f"Warning: no unique repertoire file for subject_id {subject_id}, "
                f"filename {filename}, so its counts are left empty",
                file=sys.stderr,
            )
    paths = list(dict.fromkeys(path for path in row_paths if path is not None))

    # Process several files at once, each in its own process
    sizes = {path: path.stat().st_size for path in paths}
    total_bytes = sum(sizes.values())
    done_bytes = 0
    start = time.monotonic()
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = executor.map(count_sequences, paths)
        for i, (path, (sequences, clean)) in enumerate(zip(paths, results), start=1):
            counts[path] = (sequences, clean)
            if clean is None:
                print(
                    f"Warning: {path.name} is missing a column from {', '.join(CLEAN_COLUMNS)}, "
                    "so its clean_count is left empty",
                    file=sys.stderr,
                )
            done_bytes += sizes[path]
            minutes = (time.monotonic() - start) / 60
            minutes_left = minutes * (total_bytes - done_bytes) / done_bytes if done_bytes else 0
            clean_text = "" if clean is None else f", {clean:,} clean"
            print(
                f"[{i}/{len(paths)}] {path.name}: {sequences:,} sequences{clean_text} | "
                f"{done_bytes / 1e9:.1f} of {total_bytes / 1e9:.1f} GB done, "
                f"about {minutes_left:.0f} min left",
                flush=True,
            )

    # Write the copy of the metadata with the new columns at the end
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames + ["sequence_count", "clean_count"], lineterminator="\n"
        )
        writer.writeheader()
        for row, path in zip(rows, row_paths):
            sequences, clean = counts.get(path, ("", None))
            row["sequence_count"] = sequences
            row["clean_count"] = "" if clean is None else clean
            writer.writerow(row)

    minutes = (time.monotonic() - start) / 60
    print(
        f"Wrote {output_path} in {minutes:.1f} min ({len(paths)} files counted, "
        f"{row_paths.count(None)} metadata rows without a file)"
    )

    # Report repertoire files that no metadata row points to
    used = {path.name for path in paths}
    unused = [path.name for path in tsv_files if path.name not in used]
    if unused:
        shown = ", ".join(unused[:10]) + (", ..." if len(unused) > 10 else "")
        print(f"{len(unused)} .tsv files in {args.repertoire_dir} match no metadata row: {shown}")


if __name__ == "__main__":
    main()