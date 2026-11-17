#!/usr/bin/env bash
#
# Retrain the frame-selection networks on subject-disjoint cross-validation splits.
#
# WHY
#   The 300 ACOUSLIC-AI development cases come from 281 subjects; 19 subjects were scanned twice.
#   The published folds were split by scan, so some test scans had the same subject's other scan
#   in their fold's training set.  The corrected folds keep every test set exactly as published and
#   withhold the 6-9 training/validation scans per fold that share a subject with an evaluated scan
#   (see data_preparation/make_subject_disjoint_splits.py).
#
# WHAT IT RUNS, IN ORDER (each step can be switched off)
#   1. MAKE_SPLITS=1   write the corrected split lists for both families
#   2. PREPARE=1       extract the frame images for those lists, each family with ITS OWN
#                      balanced CSV
#   3. always          verify the extracted images against the original frame folders, file by
#                      file (catches a wrong balanced CSV and leftover images of withheld scans);
#                      training does not start unless this passes
#   4. always          train every requested network that is not already in the manifest
#
# NETWORKS
#     relnet      relevant vs irrelevant.  Stage 1 of the two-step method AND, alone, "One-step".
#     optnet      optimal vs suboptimal among relevant frames (pos_only).  Stage 2.
#     optonly     optimal vs everything else (opt_only), on the opt_no_subopt family.
#     threeclass  irrelevant / optimal / suboptimal (num_classes=3).  DenseNet only in the paper.
#   "Two-step (relevant)" and the shortlist sizes K reuse relnet + optnet: no extra training.
#
# WHAT THE PAPER NEEDS: 65 networks, best run in two parts
#     1. main results     ARCHS=densenet NETS="relnet optnet optonly threeclass"   20 nets  ~8.5 h
#     2. backbone tables  ARCHS="resnet efficientnet convnext" NETS="relnet optnet optonly"
#                                                                               45 nets  ~62 h
#   Measured per network on the GPU box (opt_using_subopt / opt_no_subopt family):
#     densenet ~28/18 min   resnet ~27/20 min   efficientnet ~58/38 min   convnext ~195/125 min
#   ConvNeXt is ~43 h of part 2, so it is listed last; it is the one to stop if time runs short.
#
# USAGE (from anywhere; paths below, or override any of them from the environment)
#   first time - splits, frames, check, then the main DenseNet networks:
#     MAKE_SPLITS=1 PREPARE=1 ARCHS=densenet NETS="relnet optnet optonly threeclass" \
#       nohup ./part1_frame_classification/scripts/train_subject_disjoint.sh > train_sd_1.log 2>&1 &
#   then the backbones:
#     ARCHS="resnet efficientnet convnext" \
#       nohup ./part1_frame_classification/scripts/train_subject_disjoint.sh > train_sd_2.log 2>&1 &
#
#   DRY_RUN=1 runs steps 1 and 3 for real (both are cheap and write nothing that matters) but
#   only PRINTS the extraction and training commands, with a time estimate.
#   RESUMABLE: finished runs are appended to $MANIFEST and skipped when re-run.  Delete a line to
#   force that run again.  The manifest is also what the evaluation scripts read.
#
# !! CONFIRM BEFORE TRAINING threeclass !!
#   The configs of the published three-class runs (206-210) were not available when this was
#   written, so THREECLASS_LOSS below is a guess.  Read "loss" from 206/config.json in LOG_DIR
#   and set it.  Every other hyper-parameter is copied from the saved configs of runs
#   170-174 / 196-200 / 201-205.
#
set -u -o pipefail

# ----------------------------------------------------------------------------- CONFIG --------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT="${OUT:-/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output}"

ARCHS="${ARCHS:-densenet}"                       # densenet resnet efficientnet convnext
NETS="${NETS:-relnet optnet optonly}"            # add threeclass once its loss is confirmed
FOLDS="${FOLDS:-0 1 2 3 4}"

# the ORIGINAL (published) families: their data_split lists are the source of the correction,
# and their images are the reference the new extraction is checked against
ORIGINAL_USING_SUBOPT="${ORIGINAL_USING_SUBOPT:-$OUT/cross_valid_folds_opt_using_subopt}"
ORIGINAL_NO_SUBOPT="${ORIGINAL_NO_SUBOPT:-$OUT/cross_valid_folds_opt_no_subopt}"
# the corrected families: split lists first, then the extracted frames, in the same folder
CORRECTED_USING_SUBOPT="${CORRECTED_USING_SUBOPT:-$OUT/cross_valid_folds_opt_using_subopt_subjdisjoint}"
CORRECTED_NO_SUBOPT="${CORRECTED_NO_SUBOPT:-$OUT/cross_valid_folds_opt_no_subopt_subjdisjoint}"

# step 1: the released per-sweep circumference file, which carries subject_id
MAKE_SPLITS="${MAKE_SPLITS:-0}"
SUBJECT_CSV="${SUBJECT_CSV:-$OUT/fetal_abdominal_circumferences_per_sweep.csv}"
# step 2: each family was extracted from its own balanced CSV - do not swap them
PREPARE="${PREPARE:-0}"
BALANCED_CSV_USING_SUBOPT="${BALANCED_CSV_USING_SUBOPT:-$OUT/labels_balanced_opt_using_subopt.csv}"
BALANCED_CSV_NO_SUBOPT="${BALANCED_CSV_NO_SUBOPT:-$OUT/labels_balanced_opt_no_subopt.csv}"
MHA_DIR="${MHA_DIR:-/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound}"
# step 3
SKIP_VERIFY="${SKIP_VERIFY:-0}"

LOG_DIR="${LOG_DIR:-$OUT/network}"
MANIFEST="${MANIFEST:-$LOG_DIR/subject_disjoint_manifest.csv}"
PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"

# from the saved config.json of runs 170-174 / 196-200 / 201-205
EPOCHS=50
BATCH_SIZE=32
LR=3e-4
WEIGHT_DECAY=1e-5
PATIENCE=20
MIN_EPOCH=10
LOSS=CE
THREECLASS_LOSS="${THREECLASS_LOSS:-CE}"         # <-- unverified, see the warning above
# ------------------------------------------------------------------------- END CONFIG --------

cd "$REPO_ROOT" || { echo "cannot cd to REPO_ROOT=$REPO_ROOT"; exit 1; }
die() { echo "ERROR: $*" >&2; exit 1; }
need_file() { [ -f "$1" ] || die "$2 not found: $1"; }
need_dir()  { [ -d "$1" ] || die "$2 not found: $1"; }

# net_spec <net> -> "<corrected family> <opt_only> <pos_only> <num_classes> <loss>"
net_spec() {
    case "$1" in
        relnet)     echo "$CORRECTED_USING_SUBOPT False False 2 $LOSS" ;;
        optnet)     echo "$CORRECTED_USING_SUBOPT False True  2 $LOSS" ;;
        optonly)    echo "$CORRECTED_NO_SUBOPT    True  False 2 $LOSS" ;;
        threeclass) echo "$CORRECTED_USING_SUBOPT False False 3 $THREECLASS_LOSS" ;;
        *) return 1 ;;
    esac
}

est_minutes() {
    local arch="$1" fam="$2"
    case "$arch" in
        densenet)     [ "$fam" = "nosub" ] && echo 18  || echo 28 ;;
        resnet)       [ "$fam" = "nosub" ] && echo 20  || echo 27 ;;
        efficientnet) [ "$fam" = "nosub" ] && echo 38  || echo 58 ;;
        convnext)     [ "$fam" = "nosub" ] && echo 125 || echo 195 ;;
        *)            echo 30 ;;
    esac
}

for a in $ARCHS; do
    case "$a" in densenet|resnet|efficientnet|convnext) ;; *) die "unknown architecture: $a" ;; esac
done
for n in $NETS; do net_spec "$n" >/dev/null || die "unknown network type: $n"; done

# which corrected families the requested networks actually use
FAMILIES=""
for n in $NETS; do
    read -r fam _ <<< "$(net_spec "$n")"
    case " $FAMILIES " in *" $fam "*) ;; *) FAMILIES="$FAMILIES $fam" ;; esac
done
original_of() { [ "$1" = "$CORRECTED_NO_SUBOPT" ] && echo "$ORIGINAL_NO_SUBOPT" || echo "$ORIGINAL_USING_SUBOPT"; }
balanced_of() { [ "$1" = "$CORRECTED_NO_SUBOPT" ] && echo "$BALANCED_CSV_NO_SUBOPT" || echo "$BALANCED_CSV_USING_SUBOPT"; }

# ---------------------------------------------------------------- 1. corrected split lists ---
if [ "$MAKE_SPLITS" = "1" ]; then
    echo "=== 1. writing subject-disjoint split lists ==="
    need_file "$SUBJECT_CSV" "subject mapping (SUBJECT_CSV)"
    for fam in $FAMILIES; do
        orig="$(original_of "$fam")"
        need_dir "$orig" "original split family"
        if [ -f "$fam/0/data_split/training_ids.txt" ]; then
            echo "--- $fam already has split lists, leaving them as they are"
            continue
        fi
        $PYTHON -m part1_frame_classification.scripts.data_preparation.make_subject_disjoint_splits \
            --subject_csv "$SUBJECT_CSV" --existing_split_path "$orig" --output "$fam" \
            || die "split generation failed for $fam"
    done
fi
for fam in $FAMILIES; do
    [ -f "$fam/0/data_split/training_ids.txt" ] || die "no corrected split lists in $fam
Run with MAKE_SPLITS=1, or copy the corrected split folders there."
done

# ------------------------------------------------------------------- 2. frame extraction ----
if [ "$PREPARE" = "1" ]; then
    echo ""
    echo "=== 2. extracting frames for the corrected splits ==="
    need_dir "$MHA_DIR" "scan folder (MHA_DIR)"
    for fam in $FAMILIES; do
        csv="$(balanced_of "$fam")"
        need_file "$csv" "balanced CSV for $(basename "$fam")"
        find "$fam" -maxdepth 1 -name '.DS_Store' -delete 2>/dev/null
        # extract_frames_cross_valid_existing treats every entry of the folder as a fold
        stray=""
        for e in "$fam"/*; do [ -e "$e" ] && [ ! -d "$e" ] && stray="$stray $(basename "$e")"; done
        [ -z "$stray" ] || die "non-folder entries in $fam would be read as folds: $stray"
        echo "--- $(basename "$fam")  with  $(basename "$csv")"
        if [ "$DRY_RUN" = "1" ]; then echo "    [dry run] not extracting"; continue; fi
        $PYTHON -m part1_frame_classification.scripts.data_preparation.extract_frames_from_mha_cross_valid \
            --csv "$csv" --mha_dir "$MHA_DIR" --output "$fam" --existing_split_path "$fam" \
            || die "frame extraction failed for $fam"
    done
fi

# ------------------------------------------------------------- 3. verify before training ----
if [ "$SKIP_VERIFY" != "1" ]; then
    echo ""
    echo "=== 3. checking the extracted frames against the original extraction ==="
    for fam in $FAMILIES; do
        if [ "$DRY_RUN" = "1" ] && [ ! -d "$fam/0/train" ]; then
            echo "--- $(basename "$fam"): no frames yet (dry run), check skipped"
            continue
        fi
        echo "--- $(basename "$fam")"
        $PYTHON -m part1_frame_classification.scripts.data_preparation.verify_frames_match_splits \
            --frames_dir "$fam" --reference_frames_dir "$(original_of "$fam")" \
            || die "frame check failed for $fam - not training on it (see the report above)"
    done
else
    echo "WARNING: SKIP_VERIFY=1, the extracted frames were not checked"
fi

# ------------------------------------------------------------------------------- plan ------
mkdir -p "$LOG_DIR" || die "cannot create LOG_DIR=$LOG_DIR"
[ -f "$MANIFEST" ] || echo "run_dir,arch,network,fold,data_dir,started,finished,seconds" > "$MANIFEST"

planned=0; est_total=0
for arch in $ARCHS; do
    for net in $NETS; do
        read -r fam _ <<< "$(net_spec "$net")"
        f=other; [ "$fam" = "$CORRECTED_NO_SUBOPT" ] && f=nosub
        for fold in $FOLDS; do
            grep -q ",$arch,$net,$fold," "$MANIFEST" 2>/dev/null && continue
            planned=$((planned + 1))
            est_total=$((est_total + $(est_minutes "$arch" "$f")))
        done
    done
done
echo ""
echo "=== 4. training: $planned runs still to do over [$ARCHS] x [$NETS] x [$FOLDS]"
echo "estimated GPU time: ~$((est_total / 60))h $((est_total % 60))m"
case " $NETS " in *" threeclass "*)
    echo "NOTE: threeclass uses loss=$THREECLASS_LOSS, which is NOT verified against run 206." ;;
esac

# ---------------------------------------------------------------------------- 4. train -----
total=0; skipped=0; failed=0
start_all=$(date +%s)

for arch in $ARCHS; do
  for net in $NETS; do
    read -r fam opt_only pos_only num_classes loss <<< "$(net_spec "$net")"
    for fold in $FOLDS; do
        data_dir="$fam/$fold"
        total=$((total + 1))

        if grep -q ",$arch,$net,$fold," "$MANIFEST" 2>/dev/null; then
            echo "[skip] $arch/$net fold $fold already in $MANIFEST"
            skipped=$((skipped + 1))
            continue
        fi

        echo ""
        echo "=== $arch / $net / fold $fold  ($(date '+%F %T')) ==="
        echo "    data_dir=$data_dir opt_only=$opt_only pos_only=$pos_only num_classes=$num_classes loss=$loss"
        if [ "$DRY_RUN" = "1" ]; then echo "    [dry run] would train"; continue; fi
        [ -d "$data_dir/train" ] || die "no extracted frames in $data_dir (run with PREPARE=1)"

        # the run folder is max(existing numeric folder)+1, so note what is there before the run
        before=$(ls -1 "$LOG_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
        t0=$(date +%s); started=$(date '+%F %T')

        $PYTHON -m part1_frame_classification.scripts.train_classification \
            --data_dir "$data_dir" \
            --model_name "$arch" \
            --num_classes "$num_classes" \
            --opt_only "$opt_only" \
            --pos_only "$pos_only" \
            --loss "$loss" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --weight_decay "$WEIGHT_DECAY" \
            --patience "$PATIENCE" \
            --min_epoch "$MIN_EPOCH" \
            --apply_mixup False \
            --log_dir "$LOG_DIR"
        rc=$?

        secs=$(( $(date +%s) - t0 ))
        after=$(ls -1 "$LOG_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)

        if [ "$rc" -ne 0 ]; then
            echo "    FAILED (exit $rc) after ${secs}s - continuing with the rest"
            failed=$((failed + 1)); continue
        fi
        if [ "$after" = "$before" ] || [ ! -f "$LOG_DIR/$after/model.pt" ]; then
            echo "    WARNING: no new run folder with a model.pt in $LOG_DIR; not recording this run"
            failed=$((failed + 1)); continue
        fi

        echo "$after,$arch,$net,$fold,$data_dir,$started,$(date '+%F %T'),$secs" >> "$MANIFEST"
        echo "    done in $((secs / 60))m $((secs % 60))s  ->  run folder $after"
    done
  done
done

elapsed=$(( $(date +%s) - start_all ))
echo ""
echo "================================================================"
echo "planned $total   skipped $skipped   failed $failed"
echo "total wall clock: $((elapsed / 3600))h $(((elapsed % 3600) / 60))m"
echo "run-id mapping: $MANIFEST  (the evaluation scripts read this file)"
[ "$failed" -gt 0 ] && exit 1
exit 0
