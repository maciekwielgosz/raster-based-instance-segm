#!/usr/bin/env bash
set -euo pipefail

code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_dir="$(cd -- "${code_dir}/.." && pwd)"
project_dir="$(cd -- "${run_dir}/.." && pwd)"

python_bin="${project_dir}/.tools/miniforge3/envs/treescan/bin/python"
segmentation_script="${code_dir}/pcopw_chunks_500m_Segmentacja.py"
input_dir="${run_dir}/data_input_from_laz_for_instance"
output_dir="${run_dir}/data_output_from_laz_for_instance"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python environment not found: ${python_bin}" >&2
  exit 2
fi

exec "${python_bin}" "${segmentation_script}" \
  --input-dir "${input_dir}" \
  --output-dir "${output_dir}" \
  --independent-tiles \
  "$@"
