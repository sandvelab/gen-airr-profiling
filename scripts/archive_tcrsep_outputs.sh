#!/usr/bin/env bash
# Moves the TCRsep outputs generated with the old immuneML seeding out of the way before regenerating them.
#
#   archive          before resampling: moves the generated, resampled and postprocessed TCRsep sequences and the
#                    finished sampling runs to <output_dir>/_archive_tcrsep_seed1. The trained models and the
#                    train/test p_gens are kept.
#   clean_analyses   after postprocessing: copies the TCRsep phenotype results to the archive, then deletes the
#                    TCRsep analysis caches and the network tables that include all models, so that only TCRsep is
#                    recomputed when the analyses are rerun.
#
# Only prints what it would do unless --run is given.
# Usage: bash scripts/archive_tcrsep_outputs.sh <archive|clean_analyses> <output_dir> [--run]
set -euo pipefail

MODE=$1
ROOT=$2
DRY=echo
[ "${3:-}" = "--run" ] && DRY=
ARCHIVE=$ROOT/_archive_tcrsep_seed1
MODELS="TCRsep_overlap TCRsep_kmer"
SEQUENCE_DIRS="generated_sequences generated_compairr_sequences generated_compairr_sequences_split
  resampled_sequences_raw resampled_no_train_sequences resampled_no_train_compairr_sequences
  resampled_no_train_compairr_sequences_split generated_no_train_compairr_sequences_split
  novel_generated_compairr_sequences_split novel_unique_generated_compairr_sequences_split
  novel_generated_compairr_sequences"

case $MODE in
  archive)
    for model in $MODELS; do
      for dir in $SEQUENCE_DIRS; do
        if [ -d "$ROOT/$dir/$model" ]; then
          $DRY mkdir -p "$ARCHIVE/$dir"
          $DRY mv "$ROOT/$dir/$model" "$ARCHIVE/$dir/"
        fi
      done
      for exp_dir in "$ROOT"/sampling/exp_*; do
        if [ -d "$exp_dir/$model" ]; then
          $DRY mkdir -p "$ARCHIVE/sampling/$(basename "$exp_dir")"
          $DRY mv "$exp_dir/$model" "$ARCHIVE/sampling/$(basename "$exp_dir")/"
        fi
      done
    done
    ;;
  clean_analyses)
    for model in $MODELS; do
      if [ -d "$ROOT/analyses/phenotype/$model" ]; then
        $DRY mkdir -p "$ARCHIVE/analyses/phenotype"
        $DRY cp -r "$ROOT/analyses/phenotype/$model" "$ARCHIVE/analyses/phenotype/"
      fi
    done
    find "$ROOT/analyses" -depth -name '*TCRsep*' -exec $DRY rm -rf {} +
    [ -d "$ROOT/analyses/network" ] && find "$ROOT/analyses/network" -name '*_connectivity.tsv' -exec $DRY rm {} +
    ;;
  *)
    echo "Unknown mode $MODE, use archive or clean_analyses" >&2
    exit 1
    ;;
esac

[ -n "$DRY" ] && echo "Dry run, nothing was changed. Add --run to apply."
exit 0
