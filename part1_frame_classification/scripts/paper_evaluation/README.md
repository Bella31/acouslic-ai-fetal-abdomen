# Paper evaluation

One script per results section of the paper. Each script runs the inference it needs with
`evaluate_frame_selection.py`, skips anything already computed, and writes that section's tables
to one Excel workbook.

| Script | Paper section | Output |
|---|---|---|
| `evaluate_main_comparison.py` | Main results: accuracy (top-1/3/5) and WFSS for One-step, Opt-only, Three-class, Two-step and Two-step (relevant), DenseNet | `main_comparison.xlsx` |
| `evaluate_frame_level.py` | Frame-level classification: F1 table, precision–recall, three-class confusion matrices (Supplement) | `frame_level.xlsx` |
| `evaluate_shortlist_size.py` | Shortlist size K = 3, 5, 10: coverage, top-1/3, WFSS@1/3 (Table 2) | `shortlist_size.xlsx` |
| `evaluate_backbones.py` | Backbone ablation: One-step, Opt-only and Two-step on DenseNet, EfficientNet, ResNet, ConvNeXt (Supplement tables) | `backbones.xlsx` |
| `run_paper_evaluation.sh` | runs the four in order | |

The statistical comparisons (McNemar, bootstrap, Holm) are not part of these scripts. Every
workbook has a `per_scan` sheet (one row per scan, method and fold model, with `subject_id` when
`--subject_csv` is given) to run them from.

## What the scripts need

**A networks table**, a CSV with the columns `run_dir, arch, network, fold`, saying which trained
run folder is which network:

| network | trained as | used by |
|---|---|---|
| `relnet` | relevant vs irrelevant | One-step; stage 1 of Two-step |
| `optnet` | optimal vs suboptimal, relevant frames only (`pos_only`) | stage 2 of Two-step |
| `optonly` | optimal vs everything else (`opt_only`), `opt_no_subopt` family | Opt-only |
| `threeclass` | irrelevant / optimal / suboptimal | Three-class |

- `train_subject_disjoint.sh` writes this table as it trains (`subject_disjoint_manifest.csv` in the network folder).
- `published_networks.csv` in this folder lists the runs behind the submitted manuscript, so the
  same scripts also reproduce the published numbers.

**Data:** the full `.mha` scans (`--scan_dir`), the frame-level labels CSV with every frame
(`--labels_path`), and the list of the 113 Suboptimal-only scans (`--subopt_list`). The defaults
are the paths on the GPU machine.

## Checks before anything runs

- every run's `config.json` matches its role (backbone, classes, `opt_only`/`pos_only`, split family, fold)
- within a fold, all methods share the same test set, so the comparisons are paired
- the five test sets are disjoint
- no network was trained or validated on a Suboptimal-only scan
- with `--subject_csv`, any subject shared between a network's training data and the scans it is evaluated on is reported

When a check fails, nothing runs. When predictions are missing, or a fold covers the wrong scans,
no tables are written. `--allow_incomplete` writes them anyway, named `*_INCOMPLETE.xlsx`.

## Where results go

Predictions go in the run folder of the network that produced them, named as in the earlier
evaluations:

```
<log_dir>/<run>/results/predictions_<run>[_subopt][_all_rel | _k3 | _k5 | _k10].xlsx
```

The sections share files. For example, the DenseNet Two-step predictions are computed once and
used by the main comparison, the frame-level table and the backbone table. An existing file is
not recomputed unless `--overwrite` is passed, in which case the evaluator moves the old file to
`previous_versions/`.

Workbooks go to `--out_dir`, by default `paper_evaluation_<networks table name>/` next to the
networks table. A workbook that already exists is moved to `previous_versions/` first. Per-run
logs are kept under `logs/`.

Numbers are averaged over the scans of each fold model, then over the five models. This is how
`summarize_results.py` produced the published tables. SDs are taken over the five fold means. The
frame-level Optimal-included values come from confusion matrices summed over all scans, as in the
published F1 table.

## Running

From the repository root:

```bash
# everything, for the subject-disjoint networks
nohup ./part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.sh > eval.log 2>&1 &

# one section
python -m part1_frame_classification.scripts.paper_evaluation.evaluate_shortlist_size \
    --networks <network folder>/subject_disjoint_manifest.csv

# what would run, with a time estimate
DRY_RUN=1 ./part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.sh

# tables only, from predictions that already exist (no GPU, no scans needed)
python -m part1_frame_classification.scripts.paper_evaluation.evaluate_main_comparison \
    --networks ... --summarize_only
```

Approximate inference time per scan: about 3 s for DenseNet, ResNet and EfficientNet, and 9 s for
ConvNeXt. Each fold model sees 28 Optimal-included and 113 Suboptimal-only scans.
