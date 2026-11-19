# Paper evaluation

One script per results section of the paper. Each script runs the inference it needs with
`evaluate_frame_selection.py`, skips anything already computed, and writes that section's tables
to one Excel workbook.

| Script | Paper section | Output |
|---|---|---|
| `evaluate_main_comparison.py` | Main results: accuracy (top-1/3/5) and WFSS for One-step, Opt-only, Three-class, Two-step and Two-step (relevant), DenseNet | `main_comparison.xlsx` |
| `evaluate_statistics.py` | Two-step vs Opt-only tests: exact McNemar, bootstrap CIs and p-values, Holm (Supplement table, asterisks of the main table); also resampling subjects | `statistics.xlsx` |
| `evaluate_frame_level.py` | Frame-level classification: F1 table, precision–recall, three-class confusion matrices (Supplement) | `frame_level.xlsx` |
| `make_figures.py` | Fig. 2, Fig. 3, Fig. S1 (three-class confusion matrices), Fig. S2 (WFSS distributions), drawn with the plotting functions in `evaluation_plots/` from `main_comparison.xlsx` and `frame_level.xlsx`; requires pandas >= 0.25 | `figures/` |
| `evaluate_shortlist_size.py` | Shortlist size K = 3, 5, 10: coverage, top-1/3, WFSS@1/3 (Table 2) | `shortlist_size.xlsx` |
| `evaluate_backbones.py` | Backbone ablation: One-step, Opt-only and Two-step on DenseNet, EfficientNet, ResNet, ConvNeXt (Supplement tables) | `backbones.xlsx` |
| `run_paper_evaluation.py` | runs the sections in order | |

`evaluate_statistics.py` reads the `per_scan` sheet of `main_comparison.xlsx` (one row per scan,
method and fold model, with `subject_id`). Its `paper` sheet holds the four comparisons resampling
scans, and its `by_subject` sheet repeats them resampling subjects, since some subjects contribute
two scans. The resampling is seeded and drawn in a fixed order, so a given input always gives the
same output. `--check_published` compares the `paper` sheet with the reference results of the
networks in `published_networks.csv`.

## What the scripts need

**A networks table**, a CSV with the columns `run_dir, arch, network, fold`, saying which trained
run folder is which network:

| network | trained as | used by |
|---|---|---|
| `relnet` | relevant vs irrelevant | One-step; stage 1 of Two-step |
| `optnet` | optimal vs suboptimal (`pos_only`); see below for the irrelevant frames | stage 2 of Two-step |
| `optonly` | optimal vs everything else (`opt_only`), `opt_no_subopt` family | Opt-only |
| `threeclass` | irrelevant / optimal / suboptimal | Three-class |

- `train_subject_disjoint.py` writes this table as it trains (`subject_disjoint_manifest.csv` in the network folder).
- `published_networks.csv` in this folder lists reference networks trained on per-scan splits.
  Their splits share subjects, so they are evaluated with `--allow_subject_overlap`.

**OptNet and the irrelevant frames.** The frame folders hold irrelevant, optimal and suboptimal
frames. OptNet (`pos_only`) is trained on optimal (1) vs not optimal (0), where class 0 is the
suboptimal and the irrelevant frames together: the shortlist it re-ranks at inference can contain
irrelevant frames, and they must rank as not optimal. `--exclude_irrelevant true` (not used) would
drop them. Each run saves its choice in `config.json` (a missing entry means irrelevant frames are
included) and the images it was trained on in `class_counts.csv`. The checks sheet says which
OptNet a table holds, and one table may not mix the two.

**Data:** the full `.mha` scans (`--scan_dir`), the frame-level labels CSV with every frame
(`--labels_path`), and the list of the 113 Suboptimal-only scans (`--subopt_list`). Default paths
are set at the top of `common.py`.

## Checks before anything runs

- every run's `config.json` matches its role (backbone, classes, `opt_only`/`pos_only`, split family, fold)
- within a fold, all methods share the same test set, so the comparisons are paired
- the five test sets are disjoint
- no network was trained or validated on a Suboptimal-only scan
- subjects (`--subject_csv`, required): no subject may be shared between a network's training or
  validation scans and the scans it is evaluated on, or between its training and validation scans,
  and every scan needs a subject_id. Any of these stops the run. The networks in
  `published_networks.csv` were trained on per-scan splits that share subjects; evaluate them with
  `--allow_subject_overlap`
  (also an option of `run_paper_evaluation.py`), which lists the overlap in the checks sheet instead

When a check fails, nothing runs. When predictions are missing, a fold covers the wrong scans, or a
metric is missing for some scans of a fold (so its mean would use fewer scans than the other
methods), no tables are written. `--allow_incomplete` writes them anyway, named `*_INCOMPLETE.xlsx`.

## Rules for edge cases

- **Two-step (relevant)** gives OptNet every frame RelNet calls relevant. When fewer than k frames
  are relevant, the top-k list is completed with RelNet's best remaining frames. When no frame is
  relevant, OptNet gets RelNet's single best frame, so the scan still has a selection, and the
  frame-level predictions stay RelNet's. The per-scan columns `second_stage_candidates` and
  `no_relevant_frame` show where this happened.
- **Ties.** Frames with exactly the same score (common at 1.000) are ranked in the order they were
  given: by frame number for RelNet, Opt-only and Three-class, and in RelNet's order for OptNet. The
  top-1 frame is always the first frame of the ranking, so it is also first in every top-k list.
- **F1** of a scan with target frames but no true positive is 0, not undefined. A predictions file
  may leave it empty, and a mean that skips those scans is too high, so the frame-level script
  recomputes the per-scan F1 from precision and recall. The pooled Optimal-included values are not
  affected.

## Where results go

Predictions go in the run folder of the network that produced them:

```
<log_dir>/<run>/results/predictions_<run>[_subopt][_all_rel | _k3 | _k5 | _k10].xlsx
```

The sections share files. For example, the DenseNet Two-step predictions are computed once and
used by the main comparison, the frame-level table and the backbone table. An existing file is
not recomputed unless `--overwrite` is passed, in which case the evaluator moves the existing file
to `previous_versions/`.

Workbooks go to `--out_dir`, by default `paper_evaluation_<networks table name>/` next to the
networks table. A workbook that already exists is moved to `previous_versions/` first. Per-run
logs are kept under `logs/`.

Numbers are averaged over the scans of each fold model, then over the five models. SDs are taken
over the five fold means. The frame-level Optimal-included values come from confusion matrices
summed over all scans.

## Running

From the repository root:

```bash
# DenseNet sections for the subject-disjoint networks (main table, F1 table, Table 2)
python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --sections main_comparison frame_level shortlist_size --dry_run   # plan and time estimate
nohup python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --sections main_comparison frame_level shortlist_size > <output folder>/eval_densenet.log 2>&1 &

# backbone tables, once all backbones are trained
nohup python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --sections backbones > <output folder>/eval_backbones.log 2>&1 &

# the reference networks in published_networks.csv (their splits share subjects, so the overlap is reported, not blocking)
python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --networks part1_frame_classification/scripts/paper_evaluation/published_networks.csv --allow_subject_overlap

# tables only, from predictions that already exist (no GPU, no scans needed)
python part1_frame_classification/scripts/paper_evaluation/run_paper_evaluation.py --summarize_only
```

`--help` lists the paths; default paths are set at the top of `run_paper_evaluation.py` and
`common.py`. Each section can also be run on its own,
e.g. `python -m part1_frame_classification.scripts.paper_evaluation.evaluate_shortlist_size --networks ...`.

Approximate inference time per scan: about 3 s for DenseNet, ResNet and EfficientNet, and 9 s for
ConvNeXt. Each fold model sees 28 Optimal-included and 113 Suboptimal-only scans.
