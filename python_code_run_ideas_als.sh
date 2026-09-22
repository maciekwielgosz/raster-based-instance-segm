#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${RUN_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.tools/miniforge3/envs/treescan/bin/python"

INPUT_DIR="${PROJECT_DIR}/ideas_als"
PREPARED_DIR="${RUN_DIR}/data_input_from_laz_ideas_als_dev"
STUDY_DIR="${RUN_DIR}/optimization_ideas_als_dev_weighted_pq_v1"

"${PYTHON}" "${SCRIPT_DIR}/for_instance_to_chm_gt.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${PREPARED_DIR}" \
  --split dev \
  --tree-point-mode instance \
  --chm-point-mode all-nonground \
  --dtm-mode auto \
  --dataset-name IDEAS-ALS \
  --manifest-name ideas_als_file_manifest.csv

"${PYTHON}" "${SCRIPT_DIR}/optimize_segmentation_hyperparameters.py" \
  --input-dir "${PREPARED_DIR}" \
  --gt-dir "${PREPARED_DIR}" \
  --study-dir "${STUDY_DIR}" \
  --trials 40 \
  --initial-trials 12 \
  --validation-candidates 5 \
  --objective weighted-pq \
  --search-profile broad \
  --seed 20260921

"${PYTHON}" "${SCRIPT_DIR}/evaluate_best_parameters.py" \
  --best-parameters "${STUDY_DIR}/best_parameters.json" \
  --input-dir "${RUN_DIR}/optimization_for_instance_dev_weighted_pq_v2_refined/dev_input" \
  --gt-dir "${RUN_DIR}/optimization_for_instance_dev_weighted_pq_v2_refined/dev_input" \
  --output-dir "${RUN_DIR}/evaluation_for_instance_ideas_als_opt_v1"
