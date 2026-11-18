#!/usr/bin/env bash
#
# Run every evaluation section of the paper, in order, for one set of trained networks.
#
#   1. evaluate_main_comparison   main results table + WFSS (DenseNet, five methods)
#   2. evaluate_frame_level       F1 table, precision-recall, confusion matrices (no new inference)
#   3. evaluate_shortlist_size    Table 2, shortlist sizes K = 3, 5, 10
#   4. evaluate_backbones         Supplement ablation tables, four backbones
#
# Each section runs only the evaluations whose predictions files are missing, then writes its
# workbook to OUT_DIR.  A section that fails does not stop the next one; the script exits 1 at the
# end if any did.  Re-running continues where it stopped.
#
# USAGE
#   after training (subject-disjoint networks, the default):
#     nohup ./part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.sh > eval.log 2>&1 &
#   while ConvNeXt is still training:
#     BACKBONES="densenet efficientnet resnet" ./.../run_paper_evaluation.sh
#   the published networks, to reproduce the submitted numbers (their splits share subjects
#   between training and evaluation, so the overlap is reported instead of stopping the run):
#     NETWORKS=part1_frame_classification/scripts/paper_evaluation/published_networks.csv \
#       ALLOW_SUBJECT_OVERLAP=1 ./.../run_paper_evaluation.sh
#   see what would run, with a time estimate:
#     DRY_RUN=1 ./.../run_paper_evaluation.sh
#
# SUBJECT_CSV (the released per-sweep circumference file) is required: a subject shared between a
# network's training/validation scans and the scans it is evaluated on stops the run.
#
set -u -o pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT="${OUT:-/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output}"
LOG_DIR="${LOG_DIR:-$OUT/network}"
NETWORKS="${NETWORKS:-$LOG_DIR/subject_disjoint_manifest.csv}"
OUT_DIR="${OUT_DIR:-}"                              # default: next to NETWORKS
SCAN_DIR="${SCAN_DIR:-/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound}"
LABELS="${LABELS:-$OUT/labels.csv}"
SUBOPT_LIST="${SUBOPT_LIST:-$OUT/subopt_cases.txt}"
SUBJECT_CSV="${SUBJECT_CSV:-$OUT/fetal_abdominal_circumferences_per_sweep.csv}"
BACKBONES="${BACKBONES:-densenet efficientnet resnet convnext}"
SECTIONS="${SECTIONS:-main_comparison frame_level shortlist_size backbones}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_SUBJECT_OVERLAP="${ALLOW_SUBJECT_OVERLAP:-0}"

cd "$REPO_ROOT" || { echo "cannot cd to $REPO_ROOT"; exit 1; }
[ -f "$NETWORKS" ] || { echo "networks table not found: $NETWORKS"; exit 1; }
[ -f "$SUBJECT_CSV" ] || { echo "subject mapping not found: $SUBJECT_CSV (set SUBJECT_CSV)"; exit 1; }

common=(--networks "$NETWORKS" --log_dir "$LOG_DIR" --scan_dir "$SCAN_DIR" --labels_path "$LABELS"
        --subopt_list "$SUBOPT_LIST" --python "$PYTHON")
[ -n "$OUT_DIR" ] && common+=(--out_dir "$OUT_DIR")
# only when the split lists are not at the data_dir recorded in each run's config.json
[ -n "${SPLITS_ROOT:-}" ] && common+=(--splits_root "$SPLITS_ROOT")
common+=(--subject_csv "$SUBJECT_CSV")
[ "$ALLOW_SUBJECT_OVERLAP" = "1" ] && common+=(--allow_subject_overlap)
[ "$DRY_RUN" = "1" ] && common+=(--dry_run)

status=0
for section in $SECTIONS; do
    extra=()
    [ "$section" = "backbones" ] && extra=(--archs $BACKBONES)
    echo ""
    echo "################ $section  ($(date '+%F %T'))"
    $PYTHON -m "part1_frame_classification.scripts.paper_evaluation.evaluate_$section" \
        "${common[@]}" ${extra[@]+"${extra[@]}"} || { echo "!! $section did not finish cleanly"; status=1; }
done
echo ""
echo "done ($(date '+%F %T')), exit status $status"
exit $status
