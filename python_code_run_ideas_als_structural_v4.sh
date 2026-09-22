#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${RUN_DIR}/../.tools/miniforge3/envs/treescan/bin/python" \
  "${SCRIPT_DIR}/evaluate_structural_ensemble.py" \
  --reference-dir "${RUN_DIR}/data_input_from_laz_ideas_als_dev" \
  --input-dir "${RUN_DIR}/optimization_for_instance_dev_weighted_pq_v2_refined/dev_input" \
  --gt-dir "${RUN_DIR}/optimization_for_instance_dev_weighted_pq_v2_refined/dev_input" \
  --output-dir "${RUN_DIR}/evaluation_for_instance_ideas_als_structural_classes_v4" \
  --parameter-file "${RUN_DIR}/optimization_ideas_als_chm_flexible_lmf_v3/tune/trial_0035_e2da01374221/result.json" \
  --parameter-file "${RUN_DIR}/optimization_ideas_als_chm_flexible_lmf_v3/tune/trial_0070_c7c4a4443870/result.json" \
  --parameter-file "${RUN_DIR}/optimization_ideas_als_chm_flexible_lmf_v3/tune/trial_0030_9ce894764583/result.json" \
  --seed 20260924
