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
`run_r/data_input`. A positive tree ID missing from the supplied tree summary
is reported and omitted rather than rasterized without an outlier limit.

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
