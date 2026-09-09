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
