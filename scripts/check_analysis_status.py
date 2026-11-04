""" Reports how far the analyses of a pipeline config got, per analysis and model, from what is on disk.

For each analysis in the config: when it last wrote a file, how many plots it has, how many files each model has (the
per-model caches, e.g. CompAIRR outputs), and which models are missing from its result tables. Within an analysis, the
models should all have the same number of files; a model with fewer is unfinished or was cleaned.

Usage: python3 scripts/check_analysis_status.py <analysis config> [--skip TCRsep] [--since '2026-11-04 18:00']
  --skip   models whose name contains this are listed separately, e.g. while they are being regenerated
  --since  also counts the plots written after this time, to see which analyses a rerun has reached
"""
import argparse
import re
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

TABLE_SIZE_LIMIT = 5_000_000  # result tables are small; larger TSVs are sequences or CompAIRR outputs

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--skip", default=None)
parser.add_argument("--since", default=None)
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
root = Path(config["output_dir"])
since = datetime.strptime(args.since, "%Y-%m-%d %H:%M").timestamp() if args.since else None


def mentions(text, model):
    return re.search(rf"(^|[^A-Za-z0-9]){re.escape(model)}([^A-Za-z0-9]|$)", text) is not None


processes = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
running = any("generic_pipeline.py" in p and Path(args.config).name in p for p in processes)
print(f"{args.config}: {'RUNNING' if running else 'not running'}")
print(f"output: {root}\n")

for analysis in config.get("analyses", []):
    name, models = analysis["name"], analysis["model_names"]
    folder = root / "analyses" / name / "_".join(analysis.get("subfolder_name", "").split())
    print(f"== {name}/{folder.name}")
    if not folder.is_dir():
        print("   not started (no folder)\n")
        continue

    files = [f for f in folder.rglob("*") if f.is_file()]
    if not files:
        print("   empty folder\n")
        continue
    last = datetime.fromtimestamp(max(f.stat().st_mtime for f in files)).strftime("%Y-%m-%d %H:%M")
    pngs = [f for f in files if f.suffix == ".png"]
    new_pngs = f", {sum(f.stat().st_mtime >= since for f in pngs)} since {args.since}" if since else ""
    print(f"   last write {last}, {len(pngs)} PNGs{new_pngs}")

    relative_paths = [str(f.relative_to(folder)) for f in files]
    counts = {model: sum(mentions(path, model) for path in relative_paths) for model in models}
    checked = [m for m in models if not (args.skip and args.skip in m)]
    skipped = [m for m in models if args.skip and args.skip in m]
    expected = max((counts[m] for m in checked), default=0)
    if expected:
        short = [f"{m} ({counts[m]})" for m in checked if counts[m] < expected]
        print(f"   files per model: {expected}" + (f"; fewer: {', '.join(short)}" if short else " for all"))
    else:
        print("   no per-model files (the results are in combined tables)")

    # The result tables are in the analysis folder or one level below (e.g. statistics/), and are not per-model
    # caches. Reading stops once every model was found, since some analyses have hundreds of thousands of files.
    found, n_tables = set(), 0
    for f, path in zip(files, relative_paths):
        if found == set(models):
            break
        if (f.suffix == ".tsv" and path.count("/") <= 1 and f.stat().st_size < TABLE_SIZE_LIMIT
                and not any(mentions(path, m) for m in models)):
            n_tables += 1
            text = f.read_text(errors="ignore")
            found.update(m for m in models if m not in found and mentions(text, m))

    if n_tables == 0:
        # Some analyses (e.g. innovation_diversity) only write plots, one subfolder per step
        png_paths = [str(f.relative_to(folder)) for f in pngs]
        subfolders = sorted({path.split("/")[0] for path in relative_paths if "/" in path})
        print("   no result tables (plots only); PNGs per subfolder: "
              + ", ".join(f"{s}: {sum(p.startswith(s + '/') for p in png_paths)}" for s in subfolders))
    else:
        not_in_tables = [m for m in checked if m not in found]
        print("   in result tables: " + (f"missing {', '.join(not_in_tables)}" if not_in_tables else "all models"))
    if skipped:
        print("   skipped: " + ", ".join(f"{m} ({counts[m]} files" + ("" if n_tables == 0 else
                                         f", {'in' if m in found else 'not in'} tables") + ")" for m in skipped))
    print()
