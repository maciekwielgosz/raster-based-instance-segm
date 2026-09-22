#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${RUN_DIR}/.." && pwd)"
PYTHON="${PROJECT_DIR}/.tools/miniforge3/envs/treescan/bin/python"

INPUT_DIR="${PROJECT_DIR}/ideas_als"
PREPARED_DIR="${RUN_DIR}/data_input_from_laz_ideas_als_dev"
MANIFEST="${PREPARED_DIR}/ideas_als_file_manifest.csv"
STUDY_DIR="${RUN_DIR}/optimization_ideas_als_source_balanced_loso_v2"
INCUMBENT="${RUN_DIR}/optimization_ideas_als_dev_weighted_pq_v1/best_parameters.json"
FOR_INSTANCE_DIR="${RUN_DIR}/optimization_for_instance_dev_weighted_pq_v2_refined/dev_input"
TRANSFER_DIR="${RUN_DIR}/evaluation_for_instance_ideas_als_source_balanced_loso_v2"

"${PYTHON}" "${SCRIPT_DIR}/for_instance_to_chm_gt.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${PREPARED_DIR}" \
  --split dev \
  --tree-point-mode instance \
  --chm-point-mode all-nonground \
  --dtm-mode auto \
  --dataset-name IDEAS-ALS \
  --manifest-name "$(basename "${MANIFEST}")"

"${PYTHON}" "${SCRIPT_DIR}/optimize_ideas_als_loso.py" \
  --input-dir "${PREPARED_DIR}" \
  --gt-dir "${PREPARED_DIR}" \
  --manifest "${MANIFEST}" \
  --study-dir "${STUDY_DIR}" \
  --incumbent "${INCUMBENT}" \
  --trials 64 \
  --initial-trials 16 \
  --seed 20260922

# FOR-instance is touched only after best_parameters.json has been frozen.
"${PYTHON}" "${SCRIPT_DIR}/evaluate_best_parameters.py" \
  --best-parameters "${STUDY_DIR}/best_parameters.json" \
  --input-dir "${FOR_INSTANCE_DIR}" \
  --gt-dir "${FOR_INSTANCE_DIR}" \
  --output-dir "${TRANSFER_DIR}"
