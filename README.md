# Raster-based tree crown segmentation

R and Python implementations of a canopy-height-model (CHM) workflow that
detects tree tops and segments individual tree crowns with marker-controlled
watershed segmentation.

## Scripts

- `pcopw_chunks_500m_Segmentacja.R` is the original implementation based on
  `lidR`, `terra`, `ForestTools`, and `sf`.
- `pcopw_chunks_500m_Segmentacja.py` is a Python port that preserves the input
  and output naming, 25 m tile buffer, tree-top filter, DBH calculation, and
  watershed behavior.

Both scripts expect this layout by default:

```text
run_r/
├── code/
├── data_input/
│   ├── 00_CHM_Full.vrt
│   └── chm_<tile-id>.tif
└── data_output/
    └── Segmentation3/
```

The VRT must be next to its source CHM tiles. Coordinates are expected to be
in EPSG:2180. Outputs are paired files named `ttops_<tile-id>.gpkg` and
`crowns_<tile-id>.gpkg`.

## Run the Python version

Required Python packages are `numpy`, `scipy`, `rasterio`, `geopandas`,
`shapely`, `pyogrio`, and `tqdm`.

From the `run_r` directory:

```bash
python code/pcopw_chunks_500m_Segmentacja.py
```

Process one tile or explicitly replace existing paired outputs:

```bash
python code/pcopw_chunks_500m_Segmentacja.py --tile 764000_197500
python code/pcopw_chunks_500m_Segmentacja.py --overwrite
```

Use `--input-dir` and `--output-dir` to override the default locations.

## Run the R version

Required R packages are `lidR`, `terra`, `fs`, `ForestTools`, `sf`, `future`,
`furrr`, and `progressr`.

From the `run_r` directory:

```bash
Rscript code/pcopw_chunks_500m_Segmentacja.R
```

Generated rasters and GeoPackages are intentionally excluded from this source
repository.

## Convert TreeScan LAZ plots to CHM rasters

`laz_to_chm_tif.py` converts the annotated LAZ plots in
`TreeScanPL10K_unzipped` into 0.5 m CHM GeoTIFFs under
`run_r/data_input_from_laz`. It reads plot centres from `plot_summary.xlsx`,
uses the supplied per-tree heights to reject Z outliers, and copies its CRS,
data type, NoData value, compression, and band layout from a reference CHM in
`run_r/data_input`. The standard LAS `point_source_id` is used as the stable
tree identifier because it matches `individual_tree_summary.csv`; the extra
TreeScan `treeID` field is only an internal re-numbering. A positive tree ID
missing from the supplied tree summary is reported and omitted rather than
rasterized without an outlier limit.

The additional Python dependency is `laspy` with its `lazrs-python` backend.
From the `run_r` directory, convert all plots with:

```bash
python code/laz_to_chm_tif.py
```

Use `--file <name>.laz` to process one plot, `--dry-run` to validate inputs
without reading point data, and `--overwrite` to explicitly replace existing
TIFFs.

After conversion, run tree-top and crown segmentation from the project root:

```bash
./python_code_run_from_laz.sh
```

The launcher passes `--independent-tiles`, because TreeScan plots are separate
30 m samples and some anonymized plot centres overlap. Existing paired
GeoPackage outputs are skipped; pass `--overwrite` to the launcher only when
replacement is intentional. Results are written below
`run_r/data_output_from_laz/Segmentation3`.

## Convert and segment FOR-instance

`for_instance_to_chm_gt.py` converts every available LAS file from the
official `FOR-instance` directory into the same CHM, overlapping crown GT, and
exclusive topmost-cell GT products used by the TreeScan workflow. Collection
names are included in output IDs to prevent filename collisions, and the
official `dev`/`test` assignment is retained in
`for_instance_file_manifest.csv`.

From the project root, generate or safely resume all input products with:

```bash
.tools/miniforge3/envs/treescan/bin/python \
  run_r/code/for_instance_to_chm_gt.py
```

Pass `--overwrite` to rebuild existing products. Segment the resulting CHMs
with unchanged baseline hyperparameters using:

```bash
run_r/code/python_code_run_for_instance.sh
```

The launcher writes predictions below
`run_r/data_output_from_laz_for_instance/Segmentation3`. The Python
segmentation workflow inherits the projected metre CRS separately from every
CHM, which is required because FOR-instance contains collections in several
coordinate reference systems.

The converter also supports FOR-instance-like datasets whose instance labels
are carried by `treeID` but whose semantic classes differ. The options
`--tree-point-mode instance`, `--chm-point-mode all-nonground`, and
`--dtm-mode auto` keep the instance GT separate from the points used to create
the CHM and select an available terrain-normalization method.

## Prepare and optimize on IDEAS-ALS

`python_code_run_ideas_als.sh` provides the isolated IDEAS-ALS flow. It uses
only the 193 files marked `dev`, writes prepared CHM/GT products to
`run_r/data_input_from_laz_ideas_als_dev`, and writes all optimization runs to
`run_r/optimization_ideas_als_dev_weighted_pq_v1`. The 359 IDEAS-ALS files
marked `test` are not read by the optimization.

From the project root:

```bash
run_r/code/python_code_run_ideas_als.sh
```

The launcher is resumable: completed preparation products and completed
optimization candidates are reused. IDEAS-ALS uses positive `treeID` values as
instances, all non-ground/non-outside points for the CHM, and automatic DTM
selection (class-2 terrain, already normalized Z, or a lower-envelope DTM).
After selecting the winner, the launcher calls `evaluate_best_parameters.py`
to freeze those parameters and evaluate them on the separate 21-plot
FOR-instance dev view. Transfer outputs are written below
`run_r/evaluation_for_instance_ideas_als_opt_v1`.

### Source-balanced LOSO v2

`python_code_run_ideas_als_loso_v2.sh` runs the more intensive IDEAS-ALS
experiment in a separate study directory. It evaluates 64 candidates, gives
each of the eight source collections equal weight, uses one random-forest
surrogate per leave-one-source-out fold, and searches a space shifted toward
smaller LMF windows. The existing IDEAS v1 winner is included as an incumbent
so the new search must improve on it rather than merely differ from it.

From the project root:

```bash
run_r/code/python_code_run_ideas_als_loso_v2.sh
```

Optimization artifacts are written to
`run_r/optimization_ideas_als_source_balanced_loso_v2`. The source-balanced
leaderboard and LOSO selections are stored in
`candidate_source_leaderboard.csv` and
`leave_one_source_out_selection.csv`. FOR-instance is not read during the
search; the frozen winner is evaluated there only at the final transfer step,
whose outputs are under
`run_r/evaluation_for_instance_ideas_als_source_balanced_loso_v2`.

### CHM and flexible-LMF v3

`python_code_run_ideas_als_chm_lmf_v3.sh` adds two optional CHM operations
(conditional focal pit filling and NoData-aware Gaussian smoothing) and a
monotonic LMF curve defined by four height/window control points. Both CHM
operations default to disabled, so legacy runs remain reproducible. The v3
optimizer evaluates 80 candidates on the same source-balanced IDEAS-ALS dev
view and retains the v2 winner as an incumbent.

```bash
run_r/code/python_code_run_ideas_als_chm_lmf_v3.sh
```

The isolated study directory is
`run_r/optimization_ideas_als_chm_flexible_lmf_v3`; its formally selected
winner is transferred to
`run_r/evaluation_for_instance_ideas_als_chm_flexible_lmf_v3_selected` only
after selection. The completed experiment also keeps an explicitly labelled
post-selection ablation of the best genuinely new flexible-LMF candidate in
`run_r/evaluation_for_instance_ideas_als_chm_flexible_lmf_v3_best_new`.

### CHM structural classes v4

`evaluate_structural_ensemble.py` fits a three-class router using only four CHM
features: canopy cover above 2 m, canopy-height median, 95th percentile, and
height standard deviation. A robust scaler and deterministic KMeans model are
fit on IDEAS-ALS dev CHMs; no GT labels are used for routing. Each target plot
is then segmented with one of three frozen parameter sets.

The completed v4 experiment selected the parameter triplet jointly from the
80 v3 candidates using the final equal-source IDEAS-ALS PQ. Its reproducible
launcher is:

```bash
run_r/code/python_code_run_ideas_als_structural_v4.sh
```

Outputs, the full routing manifest, and per-plot structural assignments are
under `run_r/evaluation_for_instance_ideas_als_structural_classes_v4`.

## Create crown ground truth from TreeScan labels

`laz_to_crown_gt.py` projects the labeled LAZ instances onto the exact grid of
each matching `chm_*.tif` and writes one crown contour per tree to a GeoPackage.
By default, each `gt_<LAZ-stem>.gpkg` is written beside its corresponding CHM
in `run_r/data_input_from_laz`, in layer `crowns_gt` and the CHM's CRS.

The exporter applies the same per-tree height outlier limits as the CHM
converter. It fills internal mask holes, removes disconnected label noise, and
stores both raw and cleaned cell counts. Crowns clipped by a plot edge remain
in the file but have `evaluation_eligible=0`; this lets assessment code exclude
them without losing the annotation. Use `--complete-only` if they should not be
written at all.

From the `run_r` directory, create all ground-truth files with:

```bash
python code/laz_to_crown_gt.py
```

Use `--file <name>.laz` to process one plot, `--dry-run` to validate all input
pairs, `--workers 3` for conservative plot-level parallel processing, and
`--overwrite` to explicitly replace existing GT GeoPackages. Each worker reads
a large point cloud, so increase this value cautiously. The resulting `treeID`
values match `individual_tree_summary.csv`, and the polygons can be used for
instance-level IoU/Dice, boundary-distance, detection precision/recall, or
quality-assessment metrics after matching predicted and reference instances.

### Create exclusive topmost-cell ground truth

`laz_to_topmost_crown_gt.py` creates a second GT representation designed to
match the single-label CHM segmentation domain. For every 0.5 m cell whose CHM
height is at least 2 m, it assigns the canonical `point_source_id` of the tree
whose accepted labeled LAZ point has the greatest normalized height in that
cell. A height tie is resolved deterministically in favour of the smaller tree
ID. Consequently every canopy cell belongs to exactly one tree and GT crowns
cannot overlap.

Both outputs are written beside the CHMs without replacing the original
overlapping `gt_*.gpkg` files:

- `topmost_gt_<LAZ-stem>.tif`: an `int32` label raster, with `-1` for CHM
  NoData, `0` for background below 2 m, and a positive tree ID for canopy.
- `topmost_gt_<LAZ-stem>.gpkg`: the same labels converted into non-overlapping
  polygons in layer `crowns_gt_topmost`.

Create all exclusive GT pairs from the `run_r` directory with:

```bash
python code/laz_to_topmost_crown_gt.py --workers 3
```

Use `--file <plot>` for one plot, `--dry-run` to validate the input pairs, and
`--overwrite` to replace existing topmost raster/vector pairs. The 2 m cutoff
is `MIN_CANOPY_HEIGHT_METRES` near the top of the script and intentionally
matches `MIN_HEIGHT` in the segmentation script.

## Evaluate crown instance segmentation

`evaluate_crown_segmentation.py` compares each predicted
`crowns_<plot>.gpkg` with the matching `gt_<plot>.gpkg`. It uses one-to-one
polygon matching that maximizes the number of matches above each IoU threshold
and then their total IoU. The default thresholds are 0.25, 0.50, and 0.75;
0.50 is the primary threshold used in the per-tile and per-object reports.

By default, GT crowns marked `evaluation_eligible=0` are excluded because they
are clipped by a plot edge. An unmatched prediction whose area overlaps one of
those ignored crowns by at least 50% is also ignored instead of being counted
as a false positive. This avoids penalizing a method for detecting a real tree
whose reference crown is incomplete.

From the `run_r` directory, evaluate all plots with:

```bash
python code/evaluate_crown_segmentation.py
```

Reports are written to `data_output_from_laz/quality_metrics`:

- `overall_metrics.csv`: dataset metrics at every requested IoU threshold.
- `per_tile_metrics.csv`: metrics for every plot at the primary threshold.
- `matched_crowns.csv`: matched IDs, IoU, Dice, area error, centroid error,
  Hausdorff distance, and sampled symmetric boundary distances.
- `unmatched_objects.csv`: false negatives, false positives, and predictions
  ignored at incomplete edge crowns.
- `evaluation_summary.json`: configuration and machine-readable overall
  results.

Reported detection metrics include precision, recall, and F1. Recognition
Quality (`RQ`) equals detection F1, Segmentation Quality (`SQ`) is mean IoU of
true-positive matches, and Panoptic Quality is `PQ = RQ * SQ`. Area precision,
recall, Dice, and IoU use the summed intersection areas of matched instances.
Aggregate area, centroid, Hausdorff, and symmetric-boundary errors are included
for matches at the primary threshold.

Use `--file <plot>` for one plot, `--primary-iou <value>` to change the main
cutoff, `--iou-thresholds <values...>` to select reported thresholds, and
`--overwrite` to replace existing reports. Use `--include-incomplete-gt` only
when clipped edge crowns should deliberately be scored.

To evaluate the exclusive topmost-cell GT instead of the original projected
GT, select its filename prefix and GeoPackage layer and use a separate report
directory:

```bash
python code/evaluate_crown_segmentation.py \
    --gt-prefix topmost_gt_ \
    --gt-layer crowns_gt_topmost \
    --output-dir data_output_from_laz/quality_metrics_topmost
```

## Optimize segmentation hyperparameters

`optimize_segmentation_hyperparameters.py` searches for segmentation settings
that maximize agreement with the exclusive `topmost_gt_` reference. It does
not modify the source rasters, GT, or the existing `Segmentation3` results.
Every candidate gets a separate output directory.

The study uses a deterministic, site-stratified 60/20/20 split:

- candidate parameters are fitted and ranked on the tuning plots;
- the best tuning candidates and the unchanged baseline are compared on the
  validation plots;
- the selected candidate is evaluated once on the untouched test plots;
- by default, the winner is finally run on the complete dataset.

The first candidate is always the current segmentation configuration. Initial
candidates use Latin-hypercube sampling. Later candidates are proposed by a
Random Forest surrogate using an upper-confidence-bound acquisition score, so
the search balances promising parameter regions with uncertain regions. The
default objective is `0.2 * PQ@0.25 + 0.6 * PQ@0.50 + 0.2 * PQ@0.75`.

The optimizer searches the median-filter size, both LMF height breakpoints,
the complete piecewise LMF window function, and watershed connectivity. The
LMF candidates are constrained to sensible positive, generally increasing
windows. `MIN_HEIGHT` remains fixed at 2 m because that is the threshold used
to construct `topmost_gt_`; use `--tune-min-height` only for a deliberate
experiment where this GT-domain mismatch is acceptable. Polygon connectivity
remains fixed at 4 because it changes vectorization rather than the watershed
labels.

The additional dependency is `scikit-learn`. From `run_r`, start the
recommended 40-trial study with:

```bash
../.tools/miniforge3/envs/treescan/bin/python \
    code/optimize_segmentation_hyperparameters.py --trials 40
```

Results are written to `data_output_from_laz/parameter_optimization`. The most
important files are:

- `study_configuration.json`: immutable data split, objective, and search
  space used by the study;
- `leaderboard.csv`: parameters and all IoU, Dice, F1, SQ, and PQ metrics for
  every completed phase;
- `best_parameters.json`: selected values, tune/validation/test/full scores,
  detailed metrics, and a ready-to-adapt segmentation command;
- per-candidate `segmentation.log`, `evaluation.log`, predictions, and metric
  reports, retained for auditability.

The study is resumable. Run the same command again to reuse completed work, or
increase `--trials` to continue searching. Before a long run, check the full
pipeline on four plots with:

```bash
../.tools/miniforge3/envs/treescan/bin/python \
    code/optimize_segmentation_hyperparameters.py \
    --trials 1 --initial-trials 1 --validation-candidates 1 \
    --max-plots 4 --skip-full-evaluation \
    --study-dir /tmp/segm-opt-smoke
```

Alternative objectives are available through `--objective pq`, `f1`,
`mean-iou`, or `area-iou`. Use a new `--study-dir` when changing the split,
objective, GT source, seed, or search space.
