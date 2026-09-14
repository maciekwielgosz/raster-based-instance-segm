#!/usr/bin/env python3
"""Convert TreeScan LAZ plots into CHM GeoTIFF rasters.

The TreeScan point clouds use local X/Y/Z coordinates and have no CRS in their
LAS headers. Plot centres are read from ``plot_summary.xlsx`` and local X/Y
coordinates are translated around those EPSG:2180 centres.

Canopy height is derived from annotated tree points (``point_source_id > 0``).
This standard LAS field contains the stable tree identifier used by
``individual_tree_summary.csv``; the extra ``treeID`` field is only an
internal re-numbering and is deliberately not used for metadata joins. Each
tree's lowest Z is treated as its zero-height reference, and the robust
``height_m`` values supplied in the CSV are used to reject high-Z annotation
outliers. The maximum normalized height in each 0.5 m cell becomes the CHM
value. Scanned cells without tree points are zero; cells without any scan
coverage are NoData.

GeoTIFF storage settings are copied from a reference ``chm_*.tif`` in
``run_r/data_input``. Existing outputs are skipped unless ``--overwrite`` is
given.

Run all plots:

    python code/laz_to_chm_tif.py

Run one plot for testing:

    python code/laz_to_chm_tif.py --file Rem_Herby_2016_0702506.laz
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

# Match the direct-interpreter behavior of the segmentation script by exposing
# GDAL/PROJ data from the active conda prefix before importing Rasterio.
_environment_prefix = Path(sys.prefix)
_proj_data = _environment_prefix / "share" / "proj"
_gdal_data = _environment_prefix / "share" / "gdal"
_gdal_plugins = _environment_prefix / "lib" / "gdalplugins"
if _proj_data.is_dir():
    os.environ.setdefault("PROJ_DATA", str(_proj_data))
if _gdal_data.is_dir():
    os.environ.setdefault("GDAL_DATA", str(_gdal_data))
if _gdal_plugins.is_dir():
    os.environ.setdefault("GDAL_DRIVER_PATH", str(_gdal_plugins))

try:
    import laspy
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from tqdm import tqdm
except ImportError as error:
    raise SystemExit(
        "Missing Python dependency. Install the converter requirements with:\n"
        "  mamba install -n treescan -c conda-forge laspy rasterio numpy tqdm\n"
        f"Original import error: {error}"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR.parent
PROJECT_DIR = RUN_DIR.parent

# =============================================================================
# USER-TUNABLE CONFIGURATION
# =============================================================================
DEFAULT_LAZ_DIR = PROJECT_DIR / "TreeScanPL10K_unzipped"
DEFAULT_OUTPUT_DIR = RUN_DIR / "data_input_from_laz"
DEFAULT_METADATA_FILENAME = "plot_summary.xlsx"
DEFAULT_TREE_SUMMARY_FILENAME = "individual_tree_summary.csv"
DEFAULT_REFERENCE_DIR = RUN_DIR / "data_input"
REFERENCE_TIF_GLOB = "chm_*.tif"

OUTPUT_PREFIX = "chm_"
OUTPUT_SUFFIX = ".tif"
# ``point_source_id`` is the original tree ID and matches the tree-summary
# CSV. The TreeScan-specific ``treeID`` extra dimension is a plot-local
# sequential re-numbering whose values can refer to different CSV rows.
TREE_ID_DIMENSION = "point_source_id"

PIXEL_SIZE_METRES = 0.5
PLOT_HALF_SIZE_METRES = 15.0
CHUNK_SIZE_POINTS = 2_000_000
OVERWRITE_EXISTING_OUTPUTS = False
HEIGHT_OUTLIER_TOLERANCE_METRES = 0.05

# Guard against corrupt IDs causing an excessive allocation. TreeScan IDs are
# small positive integers, normally numbering at most a few hundred per plot.
MAX_TREE_ID = 1_000_000
# =============================================================================
# END USER-TUNABLE CONFIGURATION
# =============================================================================


SPREADSHEET_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_REFERENCE_PATTERN = re.compile(r"([A-Z]+)")
PLOT_YEAR_PATTERN = re.compile(r"_(?:19|20)\d{2}_")


@dataclass(frozen=True)
class RasterTemplate:
    profile: dict
    dataset_tags: dict[str, str]
    band_tags: dict[str, str]
    band_description: str
    x_resolution: float
    y_resolution: float


@dataclass(frozen=True)
class ConversionResult:
    source: Path
    output: Path
    status: str
    trees: int = 0
    omitted_trees: int = 0
    covered_cells: int = 0
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TreeScan LAZ plots to CHM GeoTIFF rasters."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_LAZ_DIR,
        help="Directory containing LAZ files and plot_summary.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated chm_*.tif files.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help=(
            "Plot metadata workbook. Defaults to plot_summary.xlsx inside "
            "--input-dir."
        ),
    )
    parser.add_argument(
        "--tree-summary",
        type=Path,
        help=(
            "Per-tree height CSV. Defaults to individual_tree_summary.csv "
            "inside --input-dir."
        ),
    )
    parser.add_argument(
        "--reference-tif",
        type=Path,
        help=(
            "GeoTIFF whose storage settings will be copied. Defaults to the "
            "first chm_*.tif in run_r/data_input."
        ),
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="NAME",
        help="Convert only this LAZ filename or stem. Repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=OVERWRITE_EXISTING_OUTPUTS,
        help="Replace an existing output TIFF.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show planned outputs without reading points.",
    )
    return parser.parse_args()


def excel_column_index(cell_reference: str) -> int:
    match = CELL_REFERENCE_PATTERN.match(cell_reference)
    if match is None:
        raise ValueError(f"Invalid spreadsheet cell reference: {cell_reference}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in workbook.namelist():
        return []
    root = ElementTree.fromstring(workbook.read(path))
    namespace = {"x": SPREADSHEET_NAMESPACE}
    return [
        "".join(node.text or "" for node in item.findall(".//x:t", namespace))
        for item in root.findall("x:si", namespace)
    ]


def worksheet_path(workbook: zipfile.ZipFile, sheet_name: str) -> str:
    namespace = {
        "x": SPREADSHEET_NAMESPACE,
        "r": RELATIONSHIP_NAMESPACE,
    }
    workbook_root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook_root.findall("x:sheets/x:sheet", namespace):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(
                f"{{{RELATIONSHIP_NAMESPACE}}}id"
            )
            break
    if relationship_id is None:
        raise ValueError(f"Worksheet {sheet_name!r} not found")

    relationships_root = ElementTree.fromstring(
        workbook.read("xl/_rels/workbook.xml.rels")
    )
    target = None
    relationship_tag = f"{{{PACKAGE_RELATIONSHIP_NAMESPACE}}}Relationship"
    for relationship in relationships_root.findall(relationship_tag):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target")
            break
    if not target:
        raise ValueError(f"Relationship {relationship_id!r} has no target")

    normalized = PurePosixPath(target.lstrip("/"))
    if not str(normalized).startswith("xl/"):
        normalized = PurePosixPath("xl") / normalized
    return str(normalized)


def spreadsheet_cell_value(cell: ElementTree.Element, strings: list[str]) -> str:
    namespace = {"x": SPREADSHEET_NAMESPACE}
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(".//x:t", namespace)
        )

    value_node = cell.find("x:v", namespace)
    if value_node is None or value_node.text is None:
        return ""
    if cell_type == "s":
        return strings[int(value_node.text)]
    return value_node.text


def read_plot_centres(metadata_path: Path) -> dict[str, tuple[float, float]]:
    """Read plot_id, X, and Y without requiring an Excel Python package."""
    with zipfile.ZipFile(metadata_path) as workbook:
        strings = shared_strings(workbook)
        sheet_root = ElementTree.fromstring(
            workbook.read(worksheet_path(workbook, "plot_summary"))
        )

    namespace = {"x": SPREADSHEET_NAMESPACE}
    rows = sheet_root.findall("x:sheetData/x:row", namespace)
    if not rows:
        raise ValueError("The plot_summary worksheet is empty")

    parsed_rows: list[dict[int, str]] = []
    for row in rows:
        values: dict[int, str] = {}
        for cell in row.findall("x:c", namespace):
            reference = cell.attrib.get("r", "")
            values[excel_column_index(reference)] = spreadsheet_cell_value(
                cell, strings
            )
        parsed_rows.append(values)

    header = {value: index for index, value in parsed_rows[0].items()}
    required_columns = {"plot_id", "X", "Y"}
    missing_columns = required_columns - set(header)
    if missing_columns:
        raise ValueError(
            "Missing plot-summary columns: " + ", ".join(sorted(missing_columns))
        )

    centres: dict[str, tuple[float, float]] = {}
    for row in parsed_rows[1:]:
        plot_id = row.get(header["plot_id"], "").strip()
        if not plot_id:
            continue
        if plot_id in centres:
            raise ValueError(f"Duplicate plot_id in metadata: {plot_id}")
        centres[plot_id] = (
            float(row[header["X"]]),
            float(row[header["Y"]]),
        )
    return centres


def read_tree_heights(
    summary_path: Path,
) -> dict[str, dict[int, float]]:
    """Read robust TreeScan height limits keyed by source file and tree ID."""
    heights: dict[str, dict[int, float]] = {}
    with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required_columns = {"source_file", "treeID", "height_m"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                "Missing tree-summary columns: "
                + ", ".join(sorted(missing_columns))
            )
        for row in reader:
            source_file = row["source_file"].strip()
            tree_id = int(row["treeID"])
            height = float(row["height_m"])
            if tree_id <= 0 or not math.isfinite(height) or height < 0:
                continue
            source_heights = heights.setdefault(source_file, {})
            if tree_id in source_heights:
                raise ValueError(
                    f"Duplicate tree summary: {source_file}, treeID {tree_id}"
                )
            source_heights[tree_id] = height
    return heights


def year_insensitive_plot_key(filename: str) -> str:
    return PLOT_YEAR_PATTERN.sub("_", filename, count=1)


def match_tree_heights(
    laz_files: list[Path],
    heights_by_file: dict[str, dict[int, float]],
) -> tuple[dict[str, dict[int, float]], list[tuple[str, str]]]:
    """Resolve exact names, then unique district/plot IDs for year typos."""
    by_plot_key: dict[str, list[str]] = {}
    for summary_name in heights_by_file:
        by_plot_key.setdefault(
            year_insensitive_plot_key(summary_name), []
        ).append(summary_name)

    resolved: dict[str, dict[int, float]] = {}
    aliases: list[tuple[str, str]] = []
    for laz_path in laz_files:
        if laz_path.name in heights_by_file:
            resolved[laz_path.name] = heights_by_file[laz_path.name]
            continue
        matches = by_plot_key.get(year_insensitive_plot_key(laz_path.name), [])
        if len(matches) == 1:
            summary_name = matches[0]
            resolved[laz_path.name] = heights_by_file[summary_name]
            aliases.append((laz_path.name, summary_name))
    return resolved, aliases


def find_reference_tif(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path.resolve()
    candidates = sorted(DEFAULT_REFERENCE_DIR.glob(REFERENCE_TIF_GLOB))
    if not candidates:
        raise FileNotFoundError(
            f"No reference TIFF matching {REFERENCE_TIF_GLOB!r} in "
            f"{DEFAULT_REFERENCE_DIR}"
        )
    return candidates[0].resolve()


def load_raster_template(path: Path) -> RasterTemplate:
    with rasterio.open(path) as source:
        if source.count != 1:
            raise ValueError(f"Reference TIFF must have one band: {path}")
        x_resolution, y_resolution = source.res
        if not math.isclose(x_resolution, PIXEL_SIZE_METRES) or not math.isclose(
            y_resolution, PIXEL_SIZE_METRES
        ):
            raise ValueError(
                f"Reference resolution is {source.res}, expected "
                f"{PIXEL_SIZE_METRES} m"
            )
        if source.crs is None:
            raise ValueError(f"Reference TIFF has no CRS: {path}")
        return RasterTemplate(
            profile=source.profile.copy(),
            dataset_tags=source.tags().copy(),
            band_tags=source.tags(1).copy(),
            band_description=source.descriptions[0] or "Z",
            x_resolution=float(x_resolution),
            y_resolution=float(y_resolution),
        )


def discover_laz_files(input_dir: Path, requested: list[str]) -> list[Path]:
    all_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".laz"
    )
    if not requested:
        return all_files

    by_name = {path.name: path for path in all_files}
    by_stem = {path.stem: path for path in all_files}
    selected: list[Path] = []
    missing: list[str] = []
    for name in requested:
        match = by_name.get(name) or by_stem.get(Path(name).stem)
        if match is None:
            missing.append(name)
        elif match not in selected:
            selected.append(match)
    if missing:
        raise FileNotFoundError("LAZ files not found: " + ", ".join(missing))
    return selected


def dimension_names(reader: laspy.LasReader) -> set[str]:
    return set(reader.header.point_format.dimension_names)


def find_tree_bases(laz_path: Path) -> tuple[np.ndarray, int]:
    """First pass: find minimum Z for each positive canonical tree ID."""
    bases = np.full(1, np.inf, dtype=np.float64)
    with laspy.open(laz_path) as reader:
        if TREE_ID_DIMENSION not in dimension_names(reader):
            raise ValueError(
                f"LAZ has no {TREE_ID_DIMENSION!r} dimension: {laz_path.name}"
            )
        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            tree_ids = np.asarray(points[TREE_ID_DIMENSION], dtype=np.int64)
            z_values = np.asarray(points.z, dtype=np.float64)
            valid = (tree_ids > 0) & np.isfinite(z_values)
            if not np.any(valid):
                continue
            valid_ids = tree_ids[valid]
            largest_id = int(valid_ids.max())
            if largest_id > MAX_TREE_ID:
                raise ValueError(
                    f"Tree ID {largest_id} exceeds MAX_TREE_ID={MAX_TREE_ID}"
                )
            if largest_id >= bases.size:
                bases = np.pad(
                    bases,
                    (0, largest_id + 1 - bases.size),
                    constant_values=np.inf,
                )
            np.minimum.at(bases, valid_ids, z_values[valid])

    tree_count = int(np.count_nonzero(np.isfinite(bases[1:])))
    if tree_count == 0:
        raise ValueError(f"No points with {TREE_ID_DIMENSION} > 0")
    return bases, tree_count


def rasterize_plot(
    laz_path: Path,
    tree_bases: np.ndarray,
    tree_height_limits: np.ndarray,
    nodata_value: float,
) -> tuple[np.ndarray, int]:
    """Second pass: rasterize maximum normalized tree height and coverage."""
    plot_size = 2.0 * PLOT_HALF_SIZE_METRES
    cell_count = plot_size / PIXEL_SIZE_METRES
    rounded_cell_count = round(cell_count)
    if not math.isclose(cell_count, rounded_cell_count):
        raise ValueError(
            "2 * PLOT_HALF_SIZE_METRES must be divisible by PIXEL_SIZE_METRES"
        )
    width = height = int(rounded_cell_count)
    chm = np.zeros(width * height, dtype=np.float32)
    covered = np.zeros(width * height, dtype=bool)

    with laspy.open(laz_path) as reader:
        for points in reader.chunk_iterator(CHUNK_SIZE_POINTS):
            x_values = np.asarray(points.x, dtype=np.float64)
            y_values = np.asarray(points.y, dtype=np.float64)
            z_values = np.asarray(points.z, dtype=np.float64)
            tree_ids = np.asarray(points[TREE_ID_DIMENSION], dtype=np.int64)

            inside = (
                np.isfinite(x_values)
                & np.isfinite(y_values)
                & (x_values >= -PLOT_HALF_SIZE_METRES)
                & (x_values < PLOT_HALF_SIZE_METRES)
                & (y_values > -PLOT_HALF_SIZE_METRES)
                & (y_values <= PLOT_HALF_SIZE_METRES)
            )
            positions = np.flatnonzero(inside)
            if positions.size == 0:
                continue

            columns = np.floor(
                (x_values[positions] + PLOT_HALF_SIZE_METRES)
                / PIXEL_SIZE_METRES
            ).astype(np.int64)
            rows = np.floor(
                (PLOT_HALF_SIZE_METRES - y_values[positions])
                / PIXEL_SIZE_METRES
            ).astype(np.int64)
            rows = np.minimum(rows, height - 1)
            cell_indices = rows * width + columns
            covered[cell_indices] = True

            selected_ids = tree_ids[positions]
            tree_mask = (
                (selected_ids > 0)
                & (selected_ids < tree_bases.size)
                & np.isfinite(z_values[positions])
            )
            tree_positions = positions[tree_mask]
            if tree_positions.size == 0:
                continue
            tree_cell_indices = cell_indices[tree_mask]
            tree_ids_in_cells = tree_ids[tree_positions]
            normalized_heights = (
                z_values[tree_positions] - tree_bases[tree_ids_in_cells]
            )
            normalized_heights = np.maximum(normalized_heights, 0.0)
            below_height_limit = normalized_heights <= (
                tree_height_limits[tree_ids_in_cells]
                + HEIGHT_OUTLIER_TOLERANCE_METRES
            )
            np.maximum.at(
                chm,
                tree_cell_indices[below_height_limit],
                normalized_heights[below_height_limit].astype(np.float32),
            )

    nodata = np.float32(nodata_value)
    chm[~covered] = nodata
    return chm.reshape((height, width)), int(np.count_nonzero(covered))


def output_profile(
    template: RasterTemplate,
    raster: np.ndarray,
    centre_x: float,
    centre_y: float,
) -> dict:
    profile = template.profile.copy()
    profile.update(
        width=raster.shape[1],
        height=raster.shape[0],
        count=1,
        transform=from_origin(
            centre_x - PLOT_HALF_SIZE_METRES,
            centre_y + PLOT_HALF_SIZE_METRES,
            PIXEL_SIZE_METRES,
            PIXEL_SIZE_METRES,
        ),
        blockxsize=raster.shape[1],
        blockysize=min(int(profile.get("blockysize", 3)), raster.shape[0]),
        tiled=False,
    )
    return profile


def verify_output(path: Path, template: RasterTemplate) -> None:
    with rasterio.open(path) as dataset:
        expected_nodata = template.profile.get("nodata")
        checks = {
            "driver": dataset.driver == template.profile.get("driver"),
            "band count": dataset.count == 1,
            "dtype": dataset.dtypes[0] == template.profile.get("dtype"),
            "NoData": dataset.nodata == expected_nodata,
            "CRS": dataset.crs == template.profile.get("crs"),
            "resolution": all(
                math.isclose(value, PIXEL_SIZE_METRES) for value in dataset.res
            ),
            "compression": dataset.profile.get("compress")
            == template.profile.get("compress"),
            "interleave": dataset.profile.get("interleave")
            == template.profile.get("interleave"),
            "band description": dataset.descriptions[0]
            == template.band_description,
        }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("Output format verification failed: " + ", ".join(failures))


def write_raster_atomically(
    output_path: Path,
    raster: np.ndarray,
    centre_x: float,
    centre_y: float,
    template: RasterTemplate,
) -> None:
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid.uuid4().hex}{OUTPUT_SUFFIX}"
    )
    profile = output_profile(template, raster, centre_x, centre_y)
    nodata = float(profile["nodata"])
    valid_values = raster[raster != nodata]
    if valid_values.size == 0:
        raise ValueError("Raster contains no scanned cells")

    try:
        with rasterio.open(temporary_path, "w", **profile) as destination:
            destination.write(raster.astype(profile["dtype"], copy=False), 1)
            destination.set_band_description(1, template.band_description)
            dataset_tags = template.dataset_tags.copy()
            dataset_tags["AREA_OR_POINT"] = "Area"
            destination.update_tags(**dataset_tags)

            band_tags = {
                key: value
                for key, value in template.band_tags.items()
                if not key.startswith("STATISTICS_")
            }
            band_tags.update(
                STATISTICS_MINIMUM=format(float(valid_values.min()), ".15g"),
                STATISTICS_MAXIMUM=format(float(valid_values.max()), ".15g"),
                STATISTICS_MEAN=format(float(valid_values.mean()), ".15g"),
                STATISTICS_STDDEV=format(float(valid_values.std()), ".15g"),
            )
            destination.update_tags(1, **band_tags)

        verify_output(temporary_path, template)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def convert_one(
    laz_path: Path,
    output_dir: Path,
    centre: tuple[float, float],
    tree_heights: dict[int, float],
    template: RasterTemplate,
    overwrite: bool,
) -> ConversionResult:
    output_path = output_dir / f"{OUTPUT_PREFIX}{laz_path.stem}{OUTPUT_SUFFIX}"
    if output_path.exists() and not overwrite:
        return ConversionResult(laz_path, output_path, "skipped", detail="exists")

    try:
        tqdm.write(f"{laz_path.name}: pass 1/2 - finding tree bases")
        tree_bases, tree_count = find_tree_bases(laz_path)
        present_tree_ids = np.flatnonzero(np.isfinite(tree_bases))
        present_tree_ids = present_tree_ids[present_tree_ids > 0]
        missing_heights = [
            int(tree_id)
            for tree_id in present_tree_ids
            if int(tree_id) not in tree_heights
        ]
        if missing_heights:
            preview = ", ".join(str(tree_id) for tree_id in missing_heights[:10])
            if len(missing_heights) > 10:
                preview += ", ..."
            tqdm.write(
                f"{laz_path.name}: omitting tree IDs without height summaries: "
                f"{preview}"
            )

        tree_height_limits = np.full(
            tree_bases.shape, -np.inf, dtype=np.float64
        )
        for tree_id in present_tree_ids:
            if int(tree_id) in tree_heights:
                tree_height_limits[tree_id] = tree_heights[int(tree_id)]

        tqdm.write(f"{laz_path.name}: pass 2/2 - rasterizing CHM")
        raster, covered_cells = rasterize_plot(
            laz_path,
            tree_bases,
            tree_height_limits,
            float(template.profile["nodata"]),
        )
        write_raster_atomically(
            output_path,
            raster,
            centre[0],
            centre[1],
            template,
        )
        return ConversionResult(
            laz_path,
            output_path,
            "converted",
            trees=tree_count - len(missing_heights),
            omitted_trees=len(missing_heights),
            covered_cells=covered_cells,
        )
    except Exception as error:
        return ConversionResult(
            laz_path,
            output_path,
            "error",
            detail=f"{type(error).__name__}: {error}",
        )


def main() -> int:
    args = parse_args()
    start = time.monotonic()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    metadata_path = (
        args.metadata.resolve()
        if args.metadata is not None
        else input_dir / DEFAULT_METADATA_FILENAME
    )
    tree_summary_path = (
        args.tree_summary.resolve()
        if args.tree_summary is not None
        else input_dir / DEFAULT_TREE_SUMMARY_FILENAME
    )

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if not metadata_path.is_file():
        print(f"Plot metadata does not exist: {metadata_path}", file=sys.stderr)
        return 2
    if not tree_summary_path.is_file():
        print(f"Tree summary does not exist: {tree_summary_path}", file=sys.stderr)
        return 2

    try:
        laz_files = discover_laz_files(input_dir, args.file)
        if not laz_files:
            raise FileNotFoundError(f"No LAZ files found in {input_dir}")
        centres = read_plot_centres(metadata_path)
        heights_by_file = read_tree_heights(tree_summary_path)
        matched_heights, height_name_aliases = match_tree_heights(
            laz_files, heights_by_file
        )
        reference_path = find_reference_tif(args.reference_tif)
        template = load_raster_template(reference_path)
    except Exception as error:
        print(f"Input validation failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    missing_centres = [path.name for path in laz_files if path.name not in centres]
    if missing_centres:
        print(
            "Missing plot centres for: " + ", ".join(missing_centres),
            file=sys.stderr,
        )
        return 2

    missing_tree_summaries = [
        path.name for path in laz_files if path.name not in matched_heights
    ]
    if missing_tree_summaries:
        print(
            "Missing tree summaries for: " + ", ".join(missing_tree_summaries),
            file=sys.stderr,
        )
        return 2

    duplicate_centres = len({centres[path.name] for path in laz_files}) < len(laz_files)
    total_bytes = sum(path.stat().st_size for path in laz_files)
    print(f"LAZ files: {len(laz_files)} ({total_bytes / 1024**3:.2f} GiB)")
    print(f"Metadata: {metadata_path}")
    print(f"Tree summary: {tree_summary_path}")
    print(f"Reference TIFF: {reference_path}")
    print(f"Output directory: {output_dir}")
    if duplicate_centres:
        print(
            "Note: some anonymized plot centres overlap; unique LAZ-based "
            "output names prevent collisions."
        )
    for laz_name, summary_name in height_name_aliases:
        print(
            f"Note: using tree summary {summary_name} for {laz_name} "
            "(unique plot ID; year differs)."
        )

    if args.dry_run:
        for laz_path in laz_files:
            output_path = output_dir / f"{OUTPUT_PREFIX}{laz_path.stem}{OUTPUT_SUFFIX}"
            centre_x, centre_y = centres[laz_path.name]
            print(f"{laz_path.name} -> {output_path.name} @ ({centre_x:g}, {centre_y:g})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        convert_one(
            laz_path,
            output_dir,
            centres[laz_path.name],
            matched_heights[laz_path.name],
            template,
            args.overwrite,
        )
        for laz_path in tqdm(laz_files, unit="plot", desc="LAZ to CHM")
    ]

    converted = [result for result in results if result.status == "converted"]
    skipped = [result for result in results if result.status == "skipped"]
    errors = [result for result in results if result.status == "error"]
    print(f"Converted: {len(converted)}")
    print(f"Skipped existing: {len(skipped)}")
    if converted:
        print(f"Annotated trees represented: {sum(item.trees for item in converted)}")
        omitted_trees = sum(item.omitted_trees for item in converted)
        if omitted_trees:
            print(f"Trees omitted without height summaries: {omitted_trees}")
    if errors:
        print(f"Errors: {len(errors)}", file=sys.stderr)
        for result in errors:
            print(f"  - {result.source.name}: {result.detail}", file=sys.stderr)

    elapsed_minutes = (time.monotonic() - start) / 60.0
    print(f"Elapsed: {elapsed_minutes:.2f} min")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
