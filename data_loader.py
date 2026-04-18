"""
=============================================================================
data_loader.py — Memory-Efficient Data Loading
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
All Overture Maps parquet files are read via DuckDB spatial queries.
This is mandatory for large files (Nigeria buildings ~11 GB, Tanzania ~7 GB).
Never load these files directly into geopandas.

Key functions:
  load_buildings()   — building polygons with geometry + key attributes
  load_divisions()   — admin boundaries at specified level
  load_places()      — POIs filtered by category
  load_ghsl_raster() — clip and return GHSL raster as numpy array
  load_worldpop()    — clip and return WorldPop raster as numpy array
  get_urban_extent() — GHSL-SMOD derived urban agglomeration polygon
=============================================================================
"""

import os
import logging
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
import duckdb

from analysis_config import (
    COUNTRIES, GHSL_DATASETS, ADMIN_LEVELS_PRIORITY, ISO3_TO_ISO2,
    GHSL_BUILDUP_THRESHOLD, URBAN_SMOD_CODES,
    PLACE_CATEGORIES, LOG_DIR,
)

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    filename=str(LOG_DIR / "01_data_loader.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DUCKDB SESSION MANAGER
# ═════════════════════════════════════════════════════════════════════════════

def get_duckdb_conn() -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection with spatial extension loaded.
    Uses in-memory database — no persistence needed.
    Thread-local connections should be used in parallel contexts.
    """
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs;  LOAD httpfs;")
    # Limit memory to avoid OOM on large files (adjust based on your RAM)
    con.execute("SET memory_limit='8GB';")
    con.execute("SET threads=4;")
    return con


def _parquet_path(iso3: str, layer: str) -> Path:
    """
    Resolve the local parquet file path for a given country and layer.
    Handles the naming convention differences between Ghana and cross-country.

    Parameters
    ----------
    iso3  : ISO3 country code (e.g. "GHA", "NGA")
    layer : one of "buildings", "places", "divisions", "roads"
    """
    cfg    = COUNTRIES[iso3]
    folder = cfg["folder"]
    prefix = cfg["file_prefix"]

    # Normalise the prefix for file lookup (handle special characters)
    # The actual file on disk uses the prefix as saved by the download script
    candidates = [
        folder / f"{prefix}_{layer}.parquet",
        folder / f"{prefix.lower()}_{layer}.parquet",
        # Ghana stores files directly in the ghana folder without subfolder
        folder / f"{layer}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search by glob as fallback
    matches = list(folder.glob(f"*{layer}.parquet"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {layer}.parquet for {iso3} in {folder}\n"
        f"Expected one of: {[str(c) for c in candidates]}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# BUILDING LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_buildings(
    iso3: str,
    con: duckdb.DuckDBPyConnection = None,
    columns: list = None,
    bbox_wgs84: tuple = None,
) -> gpd.GeoDataFrame:
    """
    Load Overture Maps buildings for a country.

    Uses DuckDB to read the parquet file and optionally filter to a bbox.
    Only selected columns are loaded to minimise memory usage.

    Parameters
    ----------
    iso3       : ISO3 code
    con        : existing DuckDB connection (creates one if None)
    columns    : list of columns to load (default: id, class, subtype, geometry)
    bbox_wgs84 : (min_lon, min_lat, max_lon, max_lat) spatial filter

    Returns
    -------
    GeoDataFrame in WGS84 (EPSG:4326), reprojected to local UTM if needed.
    """
    if con is None:
        con = get_duckdb_conn()

    parquet_path = _parquet_path(iso3, "buildings")
    log.info(f"Loading buildings: {iso3} from {parquet_path}")

    if columns is None:
        # Minimal column set for BCR and BCC analysis
        # Avoids loading JSON columns (names, sources) that inflate memory
        columns = [
            "id",
            "class",
            "subtype",
            "height",
            "num_floors",
            "ST_AsWKB(geometry) AS geom_wkb",
            "bbox.xmin AS xmin",
            "bbox.ymin AS ymin",
            "bbox.xmax AS xmax",
            "bbox.ymax AS ymax",
        ]

    col_str = ", ".join(columns)

    if bbox_wgs84:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        where = (f"WHERE bbox.xmin >= {min_lon} AND bbox.ymin >= {min_lat} "
                 f"AND bbox.xmax <= {max_lon} AND bbox.ymax <= {max_lat}")
    else:
        where = ""

    query = f"""
        SELECT {col_str}
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
        {where}
    """

    print(f"  Loading buildings ({iso3}) — this may take 1–5 min for large countries ...")
    df = con.execute(query).df()

    if "geom_wkb" in df.columns:
        from shapely import wkb
        df["geometry"] = df["geom_wkb"].apply(
            lambda x: wkb.loads(bytes(x)) if x is not None else None
        )
        df = df.drop(columns=["geom_wkb"])

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[gdf.geometry.is_valid]

    print(f"  ✓ {len(gdf):,} buildings loaded ({iso3})")
    log.info(f"[OK] {len(gdf):,} buildings loaded: {iso3}")
    return gdf


def load_building_centroids(
    iso3: str,
    con: duckdb.DuckDBPyConnection = None,
    bbox_wgs84: tuple = None,
) -> gpd.GeoDataFrame:
    """
    Load building centroids only (much faster than full polygons).
    Used for BCR computation — centroid presence per 100m cell.

    Returns a GeoDataFrame of points with columns: id, class, subtype.
    """
    if con is None:
        con = get_duckdb_conn()

    parquet_path = _parquet_path(iso3, "buildings")

    if bbox_wgs84:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        where = (f"WHERE bbox.xmin >= {min_lon} AND bbox.ymin >= {min_lat} "
                 f"AND bbox.xmax <= {max_lon} AND bbox.ymax <= {max_lat}")
    else:
        where = ""

    # Compute centroid directly in DuckDB — much faster than loading polygons
    query = f"""
        SELECT
            id,
            class,
            subtype,
            height,
            num_floors,
            ST_X(ST_Centroid(geometry)) AS lon,
            ST_Y(ST_Centroid(geometry)) AS lat
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
        {where}
    """

    print(f"  Computing building centroids ({iso3}) ...")
    df = con.execute(query).df()
    df = df.dropna(subset=["lon", "lat"])

    from shapely.geometry import Point
    df["geometry"] = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print(f"  ✓ {len(gdf):,} centroids computed ({iso3})")
    return gdf


# ═════════════════════════════════════════════════════════════════════════════
# DIVISIONS LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_divisions(
    iso3: str,
    admin_level: int = None,
    con: duckdb.DuckDBPyConnection = None,
    bbox_wgs84: tuple = None,
) -> gpd.GeoDataFrame:
    """
    Load Overture Maps administrative divisions.

    Tries admin_level 8 first (sub-district), falls back to 7, then 6.
    Returns the finest available level with at least 10 units.

    Parameters
    ----------
    admin_level : specific level to request (None = auto-detect finest)
    """
    if con is None:
        con = get_duckdb_conn()

    parquet_path = _parquet_path(iso3, "divisions")

    levels_to_try = [admin_level] if admin_level else ADMIN_LEVELS_PRIORITY

    if bbox_wgs84:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        bbox_where = (f"AND bbox.xmin >= {min_lon} AND bbox.ymin >= {min_lat} "
                      f"AND bbox.xmax <= {max_lon} AND bbox.ymax <= {max_lat}")
    else:
        bbox_where = ""

    # Get ISO2 code for country filter
    # This excludes features from neighbouring countries that leak in via bbox
    iso2 = ISO3_TO_ISO2.get(iso3)

    # Per-country minimum: if level-2 count is below this, prefer NULL-level data
    # CIV and CMR have incomplete level-2 coverage in Overture Maps 2025
    min_threshold = COUNTRIES[iso3].get("min_level2_threshold", 10)
    country_filter = f"AND country = '{iso2}'" if iso2 else ""

    for level in levels_to_try:
        query = f"""
            SELECT
                id,
                names.primary           AS name,
                admin_level,
                country,
                subtype,
                class,
                division_id,
                is_territorial,
                ST_AsWKB(geometry)      AS geom_wkb
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE admin_level = {level}
            {country_filter}
            {bbox_where}
        """
        df = con.execute(query).df()

        if len(df) >= min_threshold:
            from shapely import wkb
            df["geometry"] = df["geom_wkb"].apply(
                lambda x: wkb.loads(bytes(x)) if x is not None else None
            )
            df = df.drop(columns=["geom_wkb"])
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
            gdf["admin_level_used"] = level
            print(f"  ✓ {len(gdf):,} divisions at level {level} ({iso3})")
            log.info(f"[OK] {len(gdf):,} divisions level {level}: {iso3}")
            return gdf

        print(f"  Level {level}: {len(df)} units (below threshold {min_threshold}) "
              f"— trying next level")

    # Last resort: try NULL admin_level rows filtered by country
    # These are sub-district units (communities, localities) in some countries
    if iso2:
        null_query = f"""
            SELECT
                id,
                names.primary           AS name,
                admin_level,
                country,
                subtype,
                class,
                division_id,
                is_territorial,
                ST_AsWKB(geometry)      AS geom_wkb
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE admin_level IS NULL
            AND country = '{iso2}'
            {bbox_where}
        """
        df_null = con.execute(null_query).df()
        if len(df_null) >= 10:
            from shapely import wkb
            df_null["geometry"] = df_null["geom_wkb"].apply(
                lambda x: wkb.loads(bytes(x)) if x is not None else None
            )
            df_null = df_null.drop(columns=["geom_wkb"])
            gdf = gpd.GeoDataFrame(df_null, geometry="geometry", crs="EPSG:4326")
            gdf = gdf[gdf.geometry.notnull() & gdf.geometry.is_valid]
            gdf["admin_level_used"] = "NULL"
            print(f"  ✓ {len(gdf):,} NULL-level divisions used as fallback ({iso3})")
            return gdf

    raise ValueError(
        f"No suitable admin divisions found for {iso3}. "
        f"Tried levels: {levels_to_try} + NULL fallback.\n"
        f"Available levels in file: run diagnose_divisions.py to inspect."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PLACES LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_places(
    iso3: str,
    category_filter: list = None,
    con: duckdb.DuckDBPyConnection = None,
    bbox_wgs84: tuple = None,
) -> gpd.GeoDataFrame:
    """
    Load Overture Maps places (POIs).

    Parameters
    ----------
    category_filter : list of category strings to filter on
                      (uses categories.primary field).
                      If None, loads all places.

    Returns GeoDataFrame with columns: id, name, category_primary,
    category_alternate, confidence, lon, lat.
    """
    if con is None:
        con = get_duckdb_conn()

    parquet_path = _parquet_path(iso3, "places")

    if bbox_wgs84:
        min_lon, min_lat, max_lon, max_lat = bbox_wgs84
        bbox_where = (f"AND bbox.xmin >= {min_lon} AND bbox.ymin >= {min_lat} "
                      f"AND bbox.xmax <= {max_lon} AND bbox.ymax <= {max_lat}")
    else:
        bbox_where = ""

    # Build category filter SQL
    if category_filter:
        cat_list = ", ".join(f"'{c}'" for c in category_filter)
        cat_where = f"AND categories.primary IN ({cat_list})"
    else:
        cat_where = ""

    query = f"""
        SELECT
            id,
            names.primary                          AS name,
            categories.primary                     AS category_primary,
            TRY_CAST(confidence AS DOUBLE)         AS confidence,
            ST_X(geometry)                         AS lon,
            ST_Y(geometry)                         AS lat
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
        WHERE geometry IS NOT NULL
        {bbox_where}
        {cat_where}
    """

    df = con.execute(query).df()
    df = df.dropna(subset=["lon", "lat"])

    from shapely.geometry import Point
    df["geometry"] = gpd.points_from_xy(df["lon"], df["lat"])
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print(f"  ✓ {len(gdf):,} places loaded ({iso3})"
          + (f" [category filter: {len(category_filter)} cats]"
             if category_filter else ""))
    return gdf


def load_places_by_service_category(
    iso3: str,
    con: duckdb.DuckDBPyConnection = None,
) -> dict:
    """
    Load places grouped by service category (health, education, economic, transport).
    Returns dict of {category_name: GeoDataFrame}.
    """
    results = {}
    for cat_name, cat_values in PLACE_CATEGORIES.items():
        gdf = load_places(iso3, category_filter=cat_values, con=con)
        gdf["service_category"] = cat_name
        results[cat_name] = gdf
    return results


# ═════════════════════════════════════════════════════════════════════════════
# RASTER LOADERS
# ═════════════════════════════════════════════════════════════════════════════

def _find_ghsl_raster(iso3: str, dataset: str) -> Path:
    """
    Find a clipped GHSL raster file for a country.
    Tries multiple naming conventions.
    """
    cfg    = COUNTRIES[iso3]
    folder = cfg["folder"] / "ghsl"
    prefix = iso3.lower()

    candidates = [
        folder / f"{prefix}_{dataset}.tif",
        folder / f"{iso3}_{dataset}.tif",
        folder / f"ghana_{dataset}.tif",    # Ghana special case
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: glob search
    pattern = f"*{dataset.split('_')[-1].lower()}*.tif"
    matches  = list(folder.glob(f"*{dataset}*.tif"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"GHSL {dataset} raster not found for {iso3} in {folder}\n"
        f"Run 02_ghsl.py with --scope cross_country first."
    )


def load_ghsl_raster(
    iso3: str,
    dataset: str,
    clip_geom: object = None,
) -> tuple:
    """
    Load a GHSL raster for a country.

    Parameters
    ----------
    dataset   : one of "GHS_BUILT_S", "GHS_POP", "GHS_SMOD"
    clip_geom : shapely geometry to clip to (e.g. urban agglomeration polygon)

    Returns
    -------
    (data_array, rasterio_transform, rasterio_crs, nodata_value)
    """
    raster_path = _find_ghsl_raster(iso3, dataset)

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if clip_geom is not None:
            # Reproject clip_geom to raster CRS if they differ
            # (urban extent is in WGS84; GHSL rasters are in UTM after clipping)
            clip_geom_proj = clip_geom
            if str(raster_crs).upper() not in ("EPSG:4326", "CRS84"):
                from pyproj import Transformer
                from shapely.ops import transform as shp_transform
                transformer = Transformer.from_crs(
                    "EPSG:4326", str(raster_crs), always_xy=True
                )
                clip_geom_proj = shp_transform(transformer.transform, clip_geom)

            try:
                data, transform = rio_mask(
                    src, [mapping(clip_geom_proj)], crop=True, filled=True
                )
            except Exception as mask_err:
                # If clip still fails (geometry outside raster extent),
                # load the full raster instead
                log.warning(f"Clip failed for {iso3} {dataset}: {mask_err}. Loading full raster.")
                data      = src.read()
                transform = src.transform
            crs    = raster_crs
            nodata = src.nodata
        else:
            data      = src.read()
            transform = src.transform
            crs       = raster_crs
            nodata    = src.nodata

    data = data[0] if data.ndim == 3 else data  # take first band
    log.info(f"[OK] GHSL {dataset} loaded: {iso3}, shape={data.shape}")
    return data, transform, crs, nodata


def load_worldpop_raster(
    iso3: str,
    clip_geom: object = None,
) -> tuple:
    """
    Load WorldPop population raster for a country.
    Returns (data_array, transform, crs, nodata).
    """
    cfg    = COUNTRIES[iso3]
    folder = cfg["folder"] / "worldpop"
    iso3_l = iso3.lower()

    candidates = [
        folder / f"{iso3_l}_population_2020.tif",
        folder / f"{iso3_l}_ppp_2020_UNadj.tif",
        folder / f"ghana_population_2020.tif",   # Ghana special case
    ]
    raster_path = None
    for c in candidates:
        if c.exists():
            raster_path = c
            break

    if raster_path is None:
        matches = list(folder.glob("*.tif"))
        if matches:
            raster_path = matches[0]
        else:
            raise FileNotFoundError(
                f"WorldPop raster not found for {iso3} in {folder}"
            )

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        if clip_geom is not None:
            # WorldPop is in WGS84 — clip_geom is also WGS84, so no reproject needed
            # but guard against edge cases where the raster is reprojected
            clip_geom_proj = clip_geom
            if str(raster_crs).upper() not in ("EPSG:4326", "CRS84"):
                from pyproj import Transformer
                from shapely.ops import transform as shp_transform
                transformer = Transformer.from_crs(
                    "EPSG:4326", str(raster_crs), always_xy=True
                )
                clip_geom_proj = shp_transform(transformer.transform, clip_geom)
            try:
                data, transform = rio_mask(
                    src, [mapping(clip_geom_proj)], crop=True, filled=True
                )
            except Exception:
                data      = src.read()
                transform = src.transform
            crs    = raster_crs
            nodata = src.nodata
        else:
            data      = src.read()
            transform = src.transform
            crs       = raster_crs
            nodata    = src.nodata

    data = data[0] if data.ndim == 3 else data
    return data, transform, crs, nodata


# ═════════════════════════════════════════════════════════════════════════════
# URBAN EXTENT DELINEATION
# ═════════════════════════════════════════════════════════════════════════════

def get_urban_extent(iso3: str) -> gpd.GeoDataFrame:
    """
    Derive urban agglomeration extents from GHSL-SMOD for a country.

    GHSL-SMOD urban centre (30) and dense urban cluster (23) cells
    are vectorised and dissolved into urban agglomeration polygons.

    Returns GeoDataFrame of urban extent polygons with columns:
    smod_code, area_km2, geometry (in WGS84).
    """
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    smod_data, transform, crs, nodata = load_ghsl_raster(iso3, "GHS_SMOD")

    # Mask to urban centre + dense urban cluster
    urban_mask = np.isin(smod_data, URBAN_SMOD_CODES).astype(np.uint8)

    # Vectorise
    results = []
    for geom, val in rio_shapes(urban_mask, mask=urban_mask, transform=transform):
        if val == 1:
            results.append({"geometry": shape(geom), "urban_mask": 1})

    if not results:
        raise ValueError(
            f"No urban cells found in GHSL-SMOD for {iso3}. "
            f"Check that GHS_SMOD raster contains valid data."
        )

    gdf = gpd.GeoDataFrame(results, crs=str(crs))

    # Dissolve into one urban extent polygon
    urban_union = unary_union(gdf.geometry)

    # Reproject to WGS84
    gdf_out = gpd.GeoDataFrame(
        [{"iso3": iso3, "geometry": urban_union}],
        crs=str(crs),
    )
    if str(crs) != "EPSG:4326":
        gdf_out = gdf_out.to_crs("EPSG:4326")

    # Compute area
    gdf_local = gdf_out.to_crs(COUNTRIES[iso3]["crs"])
    gdf_out["area_km2"] = gdf_local.geometry.area / 1e6

    print(f"  ✓ Urban extent: {iso3} — {gdf_out['area_km2'].sum():.1f} km²")
    return gdf_out


def check_data_availability(iso3: str) -> dict:
    """
    Check which data layers are available for a country.
    Returns a summary dict for logging and quality control.
    """
    cfg    = COUNTRIES[iso3]
    folder = cfg["folder"]
    status = {
        "iso3":     iso3,
        "name":     cfg["name"],
        "folder":   str(folder),
        "exists":   folder.exists(),
    }

    for layer in ["buildings", "places", "divisions", "roads"]:
        try:
            p = _parquet_path(iso3, layer)
            size_mb = p.stat().st_size / 1e6
            status[f"{layer}_parquet"] = f"✓ ({size_mb:.0f} MB)"
        except FileNotFoundError:
            status[f"{layer}_parquet"] = "✗ not found"

    for ds in GHSL_DATASETS.values():
        try:
            _find_ghsl_raster(iso3, ds)
            status[f"ghsl_{ds}"] = "✓"
        except FileNotFoundError:
            status[f"ghsl_{ds}"] = "✗ not found"

    wp_folder = folder / "worldpop"
    wp_files  = list(wp_folder.glob("*.tif")) if wp_folder.exists() else []
    status["worldpop"] = f"✓ ({len(wp_files)} tif)" if wp_files else "✗ not found"

    return status


def print_data_inventory():
    """Print a formatted data availability table for all countries."""
    print("\n" + "="*70)
    print("  DATA AVAILABILITY INVENTORY")
    print("="*70)
    for iso3 in COUNTRIES:
        s = check_data_availability(iso3)
        print(f"\n  {iso3} — {s['name']}")
        print(f"    Folder exists: {s['exists']}")
        for key, val in s.items():
            if key not in ("iso3", "name", "folder", "exists"):
                print(f"    {key:30s}: {val}")
    print()