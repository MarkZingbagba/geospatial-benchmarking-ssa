"""
=============================================================================
analysis_config.py — Configuration for paper's analysis
Benchmarking Open Geospatial Data Stacks for SDG 11
Sub-National Urban Monitoring in Sub-Saharan Africa

Scientific Data submission
=============================================================================
Edit BASE_DATA_DIR to match your local path before running.
=============================================================================
"""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
# UPDATE THIS to your actual project root
BASE_DATA_DIR = Path(
    r"PATH_TO_ROOT_DIRECTORY"
)

ANALYSIS_DIR  = Path(__file__).parent.parent   
OUT_DIR       = ANALYSIS_DIR / "outputs"
LOG_DIR       = ANALYSIS_DIR / "logs"
FIG_DIR       = OUT_DIR / "figures"
TABLE_DIR     = OUT_DIR / "tables"
GRID_DIR      = OUT_DIR / "grids"
GPKG_DIR      = OUT_DIR / "gpkg"

for d in [LOG_DIR, FIG_DIR, TABLE_DIR, GRID_DIR, GPKG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Study countries ──────────────────────────────────────────────────────────
# 6 SSA cross-country + Ghana (from ghana folder)
#
# bbox format: (min_lon, min_lat, max_lon, max_lat) in WGS84
# Each bbox includes a small ~0.1° buffer beyond the national boundary so
# that edge zoom-9 tiles are not missed.  The download_ms_country() function
# clips buildings back to the bbox geometry, so no cross-border leakage
# enters the merged GPKG.
COUNTRIES = {
    "GHA": {
        "name":    "Ghana",
        "folder":  BASE_DATA_DIR / "ghana",
        "crs":     "EPSG:32630",
        "cluster": "West Africa (Anglophone)",
        "capital": "Accra",
        # Overture files in ghana folder use "ghana_" prefix
        "file_prefix": "ghana",
        "min_level2_threshold": 100,  # 261 level-2 MMDAs — complete
        # WGS84 bounding box: (min_lon, min_lat, max_lon, max_lat)
        "bbox": (-3.26, 4.74, 1.19, 11.17),
    },
    "CIV": {
        "name":    "Côte d'Ivoire",
        "folder":  BASE_DATA_DIR / "cross_country" / "CIV",
        "crs":     "EPSG:32630",
        "cluster": "West Africa (Francophone)",
        "capital": "Abidjan",
        "file_prefix": "côte_divoire",   # as downloaded
        "min_level2_threshold": 200,  # only 31 level-2 units — use NULL (218 communes)
        "bbox": (-8.60, 4.36, -2.49, 10.74),
    },
    "SEN": {
        "name":    "Senegal",
        "folder":  BASE_DATA_DIR / "cross_country" / "SEN",
        "crs":     "EPSG:32628",
        "cluster": "West Africa (Francophone)",
        "capital": "Dakar",
        "file_prefix": "senegal",
        "min_level2_threshold": 40,   # 46 level-2 departments — complete
        "bbox": (-17.54, 12.31, -11.36, 16.69),
    },
    "NGA": {
        "name":    "Nigeria",
        "folder":  BASE_DATA_DIR / "cross_country" / "NGA",
        "crs":     "EPSG:32631",
        "cluster": "West Africa (Anglophone)",
        "capital": "Lagos",
        "file_prefix": "nigeria",
        "min_level2_threshold": 100,  # 773 level-2 LGAs — complete
        # Nigeria is large (~200–300 zoom-9 tiles); download will take 60–90 min
        "bbox": (2.67, 4.27, 14.68, 13.89),
    },
    "CMR": {
        "name":    "Cameroon",
        "folder":  BASE_DATA_DIR / "cross_country" / "CMR",
        "crs":     "EPSG:32632",
        "cluster": "Central Africa",
        "capital": "Douala",
        "file_prefix": "cameroon",
        "min_level2_threshold": 200,  # only 63 level-2 units — use NULL (607 sub-districts)
        "bbox": (8.50, 1.65, 16.19, 12.37),
    },
    "KEN": {
        "name":    "Kenya",
        "folder":  BASE_DATA_DIR / "cross_country" / "KEN",
        "crs":     "EPSG:32637",
        "cluster": "East Africa",
        "capital": "Nairobi",
        "file_prefix": "kenya",
        "min_level2_threshold": 100,  # 291 level-2 sub-counties — complete
        "bbox": (33.91, -4.72, 41.90, 4.62),
    },
    "TZA": {
        "name":    "Tanzania",
        "folder":  BASE_DATA_DIR / "cross_country" / "TZA",
        "crs":     "EPSG:32737",
        "cluster": "East Africa",
        "capital": "Dar es Salaam",
        "file_prefix": "tanzania",
        "min_level2_threshold": 100,  # 195 level-2 districts — complete
        # bbox covers mainland + Zanzibar archipelago
        "bbox": (29.34, -11.75, 40.44, -0.99),
    },
}

# ── GHSL file naming conventions ─────────────────────────────────────────────
# After clipping by ghsl.py, files follow: {ISO3}_GHS_{DATASET}.tif
# For Ghana the prefix is "ghana_GHS_..."
GHSL_DATASETS = {
    "BUILT_S": "GHS_BUILT_S",   # Built-up surface area 100m
    "POP":     "GHS_POP",       # Population grid 100m
    "SMOD":    "GHS_SMOD",      # Degree of urbanisation 1km
}

# GHSL built-up threshold: minimum m² per 100m cell to be classified as built-up
GHSL_BUILDUP_THRESHOLD = 10.0   # m² (excludes incidental detections)

# GHSL SMOD classification codes
SMOD_CLASSES = {
    30: "Urban Centre",
    23: "Dense Urban Cluster",
    22: "Semi-Dense Urban Cluster",
    21: "Suburban / Peri-Urban",
    13: "Rural Cluster",
    12: "Low-Density Rural",
    11: "Very Low Density Rural",
    10: "Water",
}

# Urban classes used for urban agglomeration delineation
URBAN_SMOD_CODES = [30, 23]       # Urban centre + dense urban cluster
PERIURBAN_SMOD_CODES = [22, 21]   # Semi-dense + suburban

# ── Overture Maps admin level for sub-districts ───────────────────────────────
# Level 8 = community councils / sub-metros in Ghana
# May vary by country; levels 6-8 used depending on availability
ADMIN_LEVELS_PRIORITY = [2, 3, 1]
# Overture Maps 2025 divisions schema uses:
#   Level 0 = country polygons
#   Level 1 = regions / states / provinces
#   Level 2 = districts / municipalities (Ghana's 261 MMDAs, Nigeria LGAs, etc.)
#   Level 3 = sub-districts / communities (where available)
# NULL admin_level rows = leaked features from neighbouring countries via bbox
# — excluded by the country filter in load_divisions

# ── Place categories for PCI ──────────────────────────────────────────────────
PLACE_CATEGORIES = {
    "health": [
        "hospital", "clinic", "pharmacy", "health", "healthcare",
        "doctors", "dentist", "medical_center", "dispensary",
    ],
    "education": [
        "school", "university", "college", "kindergarten",
        "primary_school", "secondary_school", "polytechnic",
    ],
    "economic": [
        "market", "bank", "financial_services", "atm",
        "money_transfer", "shopping_mall", "supermarket",
    ],
    "transport": [
        "bus_stop", "taxi", "bus_station", "ferry_terminal",
        "train_station", "transport",
    ],
}

# ── ISO3 → ISO2 mapping for Overture divisions country filter ────────────────
# The `country` field in Overture divisions uses ISO2 codes.
# This filter prevents cross-boundary contamination from bbox overlap.
ISO3_TO_ISO2 = {
    "GHA": "GH",
    "CIV": "CI",
    "SEN": "SN",
    "NGA": "NG",
    "CMR": "CM",
    "KEN": "KE",
    "TZA": "TZ",
}

# ── BCR computation parameters ────────────────────────────────────────────────
BCR_CELL_SIZE_M = 100       # 100m grid cell (matches GHSL resolution)
BCR_BUFFER_M    = 50        # point-in-cell buffer tolerance (m)

# ── PGA computation parameters ───────────────────────────────────────────────
PGA_MIN_POP_THRESHOLD = 100  # Exclude sub-districts with < 100 WorldPop persons

# ── Output file definitions (matching Data Records in paper) ─────────────────
OUTPUT_FILES = {
    "benchmark_national":    TABLE_DIR / "benchmark_metrics_national.csv",
    "benchmark_urban":       TABLE_DIR / "benchmark_metrics_urban_agglomeration.csv",
    "benchmark_subdistrict": GPKG_DIR  / "benchmark_metrics_subdistrict.gpkg",
    "suitability_matrix":    TABLE_DIR / "sdg11_suitability_matrix.csv",
    "bcr_grid_template":     GRID_DIR  / "{ISO3}_bcr_100m.tif",
}

# ── Figure settings ───────────────────────────────────────────────────────────
FIG_DPI    = 300        # Scientific Data minimum: 300 dpi
FIG_FORMAT = "pdf"      # PDF for LaTeX; change to "tif" for direct submission
FIG_STYLE  = "seaborn-v0_8-whitegrid"

# Colour scheme consistent across all figures
COLOURS = {
    "overture_only":  "#2166ac",   # blue
    "ms_only":        "#4dac26",   # green
    "both":           "#7b3294",   # purple
    "neither":        "#d1d1d1",   # light grey (data gap)
    "urban_centre":   "#d73027",   # red
    "dense_urban":    "#fc8d59",   # orange
    "semi_dense":     "#fee090",   # yellow
    "periurban":      "#4575b4",   # blue
    "rural":          "#91bfdb",   # light blue
}

# SDG 11 suitability score labels
SUITABILITY_LABELS = {
    0: "Not suitable",
    1: "Partially suitable",
    2: "Suitable (minor caveats)",
    3: "Fully suitable",
}

SUITABILITY_COLOURS = {0: "#d73027", 1: "#fee090", 2: "#a6d96a", 3: "#1a9641"}