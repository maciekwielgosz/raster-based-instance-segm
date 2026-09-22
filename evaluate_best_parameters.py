#!/usr/bin/env python3
"""Apply frozen optimized parameters to a separate CHM/GT dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import optimize_segmentation_hyperparameters as optimizer


SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTER = SCRIPT_DIR / "pcopw_chunks_500m_Segmentacja.py"
EVALUATOR = SCRIPT_DIR / "evaluate_crown_segmentation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen best_parameters.json on another dataset."
    )
    parser.add_argument("--best-parameters", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-prefix", default="topmost_gt_")
    parser.add_argument("--gt-layer", default="crowns_gt_topmost")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    best_path = args.best_parameters.resolve()
    input_dir = args.input_dir.resolve()
    gt_dir = (args.gt_dir or args.input_dir).resolve()
    output_dir = args.output_dir.resolve()

    for path, description in (
        (best_path, "best-parameters JSON"),
        (input_dir, "CHM input directory"),
        (gt_dir, "GT directory"),
        (SEGMENTER, "segmentation script"),
        (EVALUATOR, "evaluation script"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")

    payload = json.loads(best_path.read_text(encoding="utf-8"))
    parameters = optimizer.canonical_parameters(payload["parameters"])
    common = {
        "min_height",
        "median_filter_size",
        "watershed_connectivity",
        "polygon_connectivity",
    }
    preprocessing = {
        "chm_pit_fill_size",
        "chm_pit_depth",
        "chm_gaussian_sigma",
    }
    legacy_lmf = {
        "lmf_low_height_limit",
        "lmf_high_height_limit",
        "lmf_low_window_size",
        "lmf_mid_slope",
        "lmf_mid_intercept",
        "lmf_high_slope",
        "lmf_high_intercept",
    }
    flexible_lmf = {
        "lmf_control_height_1",
        "lmf_control_height_2",
        "lmf_control_height_3",
        "lmf_control_window_0",
        "lmf_control_window_1",
        "lmf_control_window_2",
        "lmf_control_window_3",
    }
    keys = set(parameters)
    allowed = common | preprocessing | legacy_lmf | flexible_lmf
    complete_lmf = legacy_lmf <= keys or flexible_lmf <= keys
    mixed_lmf = bool(keys & legacy_lmf) and bool(keys & flexible_lmf)
    if not common <= keys or not complete_lmf or mixed_lmf or keys - allowed:
        raise ValueError(
            "Invalid parameter keys: "
            f"missing common={sorted(common - keys)}, "
            f"extra={sorted(keys - allowed)}, mixed_lmf={mixed_lmf}"
        )

    segmentation_command = [
        sys.executable,
        str(SEGMENTER),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--independent-tiles",
        *optimizer.hyperparameter_cli(parameters),
    ]
    evaluation_command = [
        sys.executable,
        str(EVALUATOR),
        "--gt-dir",
        str(gt_dir),
        "--gt-prefix",
        args.gt_prefix,
        "--gt-layer",
        args.gt_layer,
        "--prediction-dir",
        str(output_dir / optimizer.PREDICTION_SUBDIRECTORY),
        "--output-dir",
        str(output_dir / "quality_metrics_topmost"),
    ]
    configuration = {
        "source_best_parameters": str(best_path),
        "source_best_parameters_sha256": optimizer.file_sha256(best_path),
        "candidate_id": payload.get("candidate_id"),
        "parameters": parameters,
        "input_directory": str(input_dir),
        "gt_directory": str(gt_dir),
        "gt_prefix": args.gt_prefix,
        "gt_layer": args.gt_layer,
        "segmentation_command": segmentation_command,
        "evaluation_command": evaluation_command,
    }
    configuration_path = output_dir / "transfer_configuration.json"
    summary = output_dir / "quality_metrics_topmost" / "evaluation_summary.json"
    if configuration_path.exists():
        existing = json.loads(configuration_path.read_text(encoding="utf-8"))
        # Version 1 stored the transient execution flag in the provenance
        # commands.  Normalize that manifest once so resume identity depends
        # only on data and parameters.
        for command_name in ("segmentation_command", "evaluation_command"):
            command = existing.get(command_name)
            if isinstance(command, list) and command[-1:] == ["--overwrite"]:
                existing[command_name] = command[:-1]
        if existing != configuration:
            raise ValueError(
                f"Existing transfer configuration differs: {configuration_path}"
            )
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            optimizer.atomic_write_json(configuration_path, configuration)
        if summary.is_file() and not args.overwrite:
            print(f"Reusing complete transfer evaluation: {summary}")
            return 0

    if args.dry_run:
        print(json.dumps(configuration, indent=2, sort_keys=True))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.atomic_write_json(configuration_path, configuration)
    segmentation_run = list(segmentation_command)
    evaluation_run = list(evaluation_command)
    if args.overwrite:
        segmentation_run.append("--overwrite")
        evaluation_run.append("--overwrite")
    run(segmentation_run)
    run(evaluation_run)
    if not summary.is_file():
        raise RuntimeError(f"Evaluation did not create {summary}")
    print(f"Transfer evaluation complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
