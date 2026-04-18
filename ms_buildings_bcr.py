"""
=============================================================================
ms_buildings_bcr.py — Microsoft ML Building Footprints: Download & BCR
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
Downloads Microsoft Global ML Building Footprints for the seven study
countries and computes MS Buildings BCR alongside Overture Maps BCR,
enabling the three-way completeness grid (Overture only / MS only /
Both / Neither) described in the Data Records section.

HOW MICROSOFT BUILDINGS ARE DISTRIBUTED
────────────────────────────────────────
Microsoft tiles the global dataset at Bing Maps quadkey zoom level 9.
Each tile is a gzipped GeoJSON file. The index CSV at:
  https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv
lists every tile quadkey and its Azure Blob download URL.

This script:
  1. Downloads the index CSV
  2. Finds all zoom-9 tiles that intersect each country's bounding box
  3. Downloads and merges tile GeoJSONs into a single country GPKG
  4. Computes building centroids via DuckDB (fast, memory-safe)
  5. Computes MS Buildings BCR relative to GHS-BUILT-S
  6. Merges MS BCR with Overture BCR into a combined completeness grid
     (values: 0=gap, 1=Overture only, 2=MS only, 3=both)

RUNNING TIME ESTIMATE
─────────────────────
Downloading tiles (once): 20–90 min per country (tile count varies)
BCR computation: 5–15 min per country

PREREQUISITES
─────────────
  pip install mercantile requests tqdm geopandas rasterio duckdb

USAGE
─────
  python 06_ms_buildings_bcr.py                  # all countries
  python 06_ms_buildings_bcr.py --country GHA    # one country
  python 06_ms_buildings_bcr.py --merge-only      # skip download, merge existing
=============================================================================
"""

import os
import csv
import gzip
import json
import logging
import argparse
import tempfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
import duckdb

import sys
sys.path.insert(0, str(Path(__file__).parent))

from analysis_config import (
    COUNTRIES, GRID_DIR, LOG_DIR, TABLE_DIR,
    GHSL_BUILDUP_THRESHOLD,
)
from data_loader import load_ghsl_raster, get_duckdb_conn

logging.basicConfig(
    filename=str(LOG_DIR / "06_ms_buildings.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── MS Buildings index URL ────────────────────────────────────────────────────
MS_INDEX_URL = (
    "https://minedbuildings.z5.web.core.windows.net/"
    "global-buildings/dataset-links.csv"
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD TILE INDEX
# ═════════════════════════════════════════════════════════════════════════════

def download_ms_index(cache_dir: Path) -> Path:
    index_path = cache_dir / "ms_buildings_index.csv"
    if index_path.exists():
        print(f"  ↳ MS index cached: {index_path}")
        return index_path

    print("  Downloading Microsoft Buildings tile index (~20 MB) ...")
    try:
        resp = requests.get(MS_INDEX_URL, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Failed to download MS Buildings index (HTTP {e.response.status_code}).\n"
            f"The hosting URL may have changed again. Check:\n"
            f"  https://github.com/microsoft/GlobalMLBuildingFootprints\n"
            f"Current URL: {MS_INDEX_URL}"
        ) from e

    total = int(resp.headers.get("content-length", 0))
    with open(index_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="ms_index.csv"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"  ✓ Index saved: {index_path}")
    return index_path

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: IDENTIFY TILES FOR A BOUNDING BOX
# ═════════════════════════════════════════════════════════════════════════════

def find_tiles_for_bbox(bbox: tuple, index_path: Path) -> dict:
    """
    Find all MS Buildings tile URLs that intersect the given bbox.

    Parameters
    ----------
    bbox       : (min_lon, min_lat, max_lon, max_lat) WGS84
    index_path : path to the downloaded index CSV

    Returns
    -------
    dict of {quadkey: url}

    HOW IT WORKS
    ────────────
    Mercantile converts the bbox to a set of zoom-9 tile quadkeys.
    The index CSV maps each quadkey to a GeoJSON download URL.
    The CSV columns are: Location, QuadKey, Url  (exact names vary
    by release — the script handles both cases).
    """
    try:
        import mercantile
    except ImportError:
        raise ImportError(
            "mercantile is required: pip install mercantile"
        )

    min_lon, min_lat, max_lon, max_lat = bbox

    # Get all zoom-9 tiles covering the bbox
    tiles    = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=9))
    quadkeys = {mercantile.quadkey(t) for t in tiles}
    print(f"  Bbox → {len(quadkeys)} zoom-9 tiles")

    # Read index CSV and match tiles
    matching = {}
    with open(index_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Normalise column names to lowercase
        for row in reader:
            row_lower = {k.lower().strip(): v for k, v in row.items()}
            qk  = row_lower.get("quadkey", "").strip()
            url = row_lower.get("url", "").strip()
            if qk in quadkeys and url:
                matching[qk] = url

    print(f"  Found {len(matching)} matching tiles in index")
    return matching


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: DOWNLOAD AND MERGE TILE FILES
# ═════════════════════════════════════════════════════════════════════════════

def download_ms_country(
    iso3: str,
    bbox: tuple,
    index_path: Path,
    output_dir: Path,
    skip_if_exists: bool = True,
) -> Path | None:
    """
    Download all MS Buildings tiles for a country and merge into a single GPKG.

    Files are streamed and parsed tile-by-tile to avoid loading the
    entire country dataset into memory simultaneously.

    Returns path to the merged GPKG, or None if download failed.
    """
    out_gpkg = output_dir / f"{iso3.lower()}_ms_buildings.gpkg"
    if out_gpkg.exists() and skip_if_exists:
        print(f"  ↳ Already exists: {out_gpkg.name}")
        return out_gpkg

    tile_urls = find_tiles_for_bbox(bbox, index_path)
    if not tile_urls:
        print(f"  ✗ No tiles found for {iso3} — check bbox and index")
        return None

    tile_dir = output_dir / f"{iso3.lower()}_tiles"
    tile_dir.mkdir(exist_ok=True)

    all_gdfs = []
    bbox_geom = __import__("shapely.geometry", fromlist=["box"]).box(*bbox)

    print(f"  Downloading {len(tile_urls)} tiles for {iso3} ...")
    for qk, url in tqdm(tile_urls.items(), desc=f"{iso3} tiles"):
        tile_path = tile_dir / f"{qk}.geojson.gz"

        # Download tile if not cached
        if not tile_path.exists():
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(tile_path, "wb") as f:
                    f.write(resp.content)
            except Exception as e:
                log.warning(f"Tile {qk} download failed: {e}")
                continue

        # Parse tile GeoJSON (gzipped)
        # Parse tile — line-delimited GeoJSON (NDJSON), one Feature per line
        try:
            features = []
            with gzip.open(tile_path, "rt", encoding="utf-8") as gz:
                for line in gz:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        features.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            if not features:
                continue
            gdf_tile = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        except Exception as e:
            log.warning(f"Tile {qk} parse failed: {e}")
            continue

        if gdf_tile.empty:
            continue

        # Clip to bbox
        gdf_tile = gdf_tile[gdf_tile.intersects(bbox_geom)].copy()
        if not gdf_tile.empty:
            all_gdfs.append(gdf_tile[["geometry"]])  # geometry only needed for BCR

    if not all_gdfs:
        print(f"  ✗ No buildings found within bbox for {iso3}")
        return None

    print(f"  Merging {len(all_gdfs)} tile(s) ...")
    merged = gpd.pd.concat(all_gdfs, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    merged = merged[merged.geometry.notnull() & merged.geometry.is_valid]

    merged.to_file(str(out_gpkg), driver="GPKG", layer="ms_buildings")
    size_mb = out_gpkg.stat().st_size / 1e6
    print(f"  ✓ Saved: {out_gpkg.name}  ({len(merged):,} buildings, {size_mb:.0f} MB)")
    log.info(f"[OK] MS Buildings {iso3}: {len(merged):,} buildings → {out_gpkg}")
    return out_gpkg


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE MS BUILDINGS BCR
# ═════════════════════════════════════════════════════════════════════════════

def compute_ms_bcr(
    iso3: str,
    ms_gpkg: Path,
    ghsl_built_s_path: Path = None,
) -> dict:
    """
    Compute BCR for Microsoft Buildings relative to GHS-BUILT-S.

    Uses the same KD-tree centroid matching as Overture Maps BCR
    (see 02_bcr_bcc_analysis.py) for direct comparability.

    Returns dict with bcr_national_pct and supporting counts.
    """
    from scipy.spatial import cKDTree

    print(f"  Computing MS Buildings BCR ({iso3}) ...")

    # Load MS building centroids from GPKG
    gdf_ms = gpd.read_file(str(ms_gpkg))
    gdf_ms = gdf_ms[gdf_ms.geometry.notnull()].copy()

    # Compute centroids in local UTM (projected CRS) to avoid UserWarning
    # about geographic CRS centroid inaccuracy, then reproject back to WGS84
    local_crs = COUNTRIES[iso3]["crs"]
    gdf_ms_proj = gdf_ms.to_crs(local_crs)
    centroids = gdf_ms_proj.copy()
    centroids["geometry"] = gdf_ms_proj.geometry.centroid
    centroids = centroids[centroids.geometry.notnull()]
    centroids = centroids.to_crs("EPSG:4326")  # back to WGS84 for CRS matching below

    # Load GHSL BUILT-S reference
    data, transform, crs, nodata = load_ghsl_raster(iso3, "GHS_BUILT_S")
    if nodata is not None:
        data = np.where(data == nodata, 0, data)
    data = np.where(data < 0, 0, data)

    buildup_mask = data > GHSL_BUILDUP_THRESHOLD
    rows, cols   = np.where(buildup_mask)

    if len(rows) == 0:
        return {"iso3": iso3, "bcr_ms_national_pct": 0, "n_ms_buildings": len(gdf_ms)}

    # Pixel centroids in raster CRS
    from rasterio.transform import xy as rio_xy
    xs, ys = rio_xy(transform, rows, cols)
    xs, ys = np.array(xs), np.array(ys)

    # Project MS centroids to raster CRS
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    transformer = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
    cx = np.array([transformer.transform(p.x, p.y)[0]
                   for p in centroids.geometry])
    cy = np.array([transformer.transform(p.x, p.y)[1]
                   for p in centroids.geometry])

    valid = np.isfinite(cx) & np.isfinite(cy)
    cx, cy = cx[valid], cy[valid]

    # KD-tree matching
    tree      = cKDTree(np.column_stack([cx, cy]))
    ghsl_pts  = np.column_stack([xs, ys])
    distances, _ = tree.query(ghsl_pts, k=1, workers=-1)

    tolerance_m = 100.0
    covered     = distances <= tolerance_m
    n_covered   = int(covered.sum())
    n_total     = len(rows)
    bcr         = round(n_covered / n_total * 100, 2) if n_total > 0 else 0

    result = {
        "iso3":                 iso3,
        "n_ms_buildings":       len(gdf_ms),
        "n_ghsl_cells":         n_total,
        "n_covered_ms":         n_covered,
        "bcr_ms_national_pct":  bcr,
    }
    print(f"  ✓ MS BCR ({iso3}): {bcr:.1f}%  ({n_covered:,}/{n_total:,} cells)")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE THREE-WAY COMPLETENESS GRID
# ═════════════════════════════════════════════════════════════════════════════

def generate_three_way_grid(
    iso3: str,
    ms_gpkg: Path,
    overture_centroids_gpkg: Path = None,
) -> Path:
    """
    Generate the three-way BCR completeness grid described in Data Records:

      Value 0 = GHSL built-up cell, NO building in either dataset (data gap)
      Value 1 = Overture Maps only
      Value 2 = Microsoft Buildings only
      Value 3 = Both datasets (Overture AND Microsoft)
      Value 255 = nodata (non-built-up cell)

    This replaces the binary (0/1) grid from save_bcr_grid() in
    02_bcr_bcc_analysis.py and produces the full Figure 2 as described
    in the paper.

    Parameters
    ----------
    iso3                  : ISO3 country code
    ms_gpkg               : path to MS Buildings GPKG (from this script)
    overture_centroids_gpkg : path to Overture centroids GPKG
                            (auto-detected from GRID_DIR if None)
    """
    from rasterio.transform import rowcol

    out_path = GRID_DIR / f"{iso3}_bcr_100m_threeway.tif"
    if out_path.exists():
        print(f"  ↳ Three-way grid exists: {out_path.name}")
        return out_path

    print(f"  Generating three-way completeness grid ({iso3}) ...")

    # Load GHSL BUILT-S
    data, transform, crs, nodata = load_ghsl_raster(iso3, "GHS_BUILT_S")
    if nodata is not None:
        data = np.where(data == nodata, 0, data)
    data = np.where(data < 0, 0, data)
    buildup_mask = (data > GHSL_BUILDUP_THRESHOLD).astype(np.uint8)

    # Initialise grid: 0 = gap (built-up but no buildings)
    grid = np.zeros_like(buildup_mask, dtype=np.uint8)

    def mark_centroids(gpkg_path: Path, bit_value: int):
        """Mark cells covered by centroids from a GPKG with bit_value."""
        try:
            gdf = gpd.read_file(str(gpkg_path))
            gdf = gdf[gdf.geometry.notnull()].copy()

            # Compute centroid in local UTM (projected CRS) to avoid
            # UserWarning about geographic CRS centroid inaccuracy
            local_crs = COUNTRIES[iso3]["crs"]
            gdf = gdf.to_crs(local_crs)
            gdf["geometry"] = gdf.geometry.centroid

            # Project to raster CRS
            if str(gdf.crs) != str(crs):
                gdf = gdf.to_crs(str(crs))

            xs = gdf.geometry.x.values
            ys = gdf.geometry.y.values

            rows_idx, cols_idx = rowcol(transform, xs, ys)
            valid = (
                (rows_idx >= 0) & (rows_idx < grid.shape[0]) &
                (cols_idx >= 0) & (cols_idx < grid.shape[1])
            )
            rows_idx = rows_idx[valid]
            cols_idx = cols_idx[valid]

            # Set bit_value using bitwise OR so overlapping cells get value 3
            grid[rows_idx, cols_idx] |= bit_value
            print(f"    Marked {valid.sum():,} cells with bit {bit_value}")
        except Exception as e:
            print(f"    ✗ Could not mark {gpkg_path.name}: {e}")

    # Bit 1 = Overture Maps present
    if overture_centroids_gpkg and Path(overture_centroids_gpkg).exists():
        mark_centroids(Path(overture_centroids_gpkg), bit_value=1)
    else:
        # Try to find Overture centroids from the country data folder
        cfg = COUNTRIES[iso3]
        candidates = list(cfg["folder"].glob("*buildings*.gpkg")) + \
                     list(cfg["folder"].glob("*buildings*.parquet"))
        if candidates:
            print(f"    Using Overture file: {candidates[0].name}")
            # Load via geopandas (GPKG) or DuckDB centroid (parquet)
            if candidates[0].suffix == ".gpkg":
                mark_centroids(candidates[0], bit_value=1)
            else:
                # DuckDB centroid extraction for parquet
                con = get_duckdb_conn()
                df_c = con.execute(f"""
                    SELECT ST_X(ST_Centroid(geometry)) AS cx,
                           ST_Y(ST_Centroid(geometry)) AS cy
                    FROM read_parquet('{candidates[0]}', hive_partitioning=0)
                    WHERE geometry IS NOT NULL
                """).df()
                df_c = df_c.dropna()
                from shapely.geometry import Point
                tmp_gdf = gpd.GeoDataFrame(
                    {"geometry": gpd.points_from_xy(df_c["cx"], df_c["cy"])},
                    crs="EPSG:4326"
                )
                with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tf:
                    tmp_path = tf.name
                tmp_gdf.to_file(tmp_path, driver="GPKG")
                mark_centroids(Path(tmp_path), bit_value=1)
                os.unlink(tmp_path)

    # Bit 2 = Microsoft Buildings present
    mark_centroids(ms_gpkg, bit_value=2)

    # Apply built-up mask: non-built-up cells → nodata (255)
    final_grid = np.where(buildup_mask == 1, grid, 255)

    # Write GeoTIFF
    out_meta = {
        "driver": "GTiff", "dtype": "uint8",
        "width": grid.shape[1], "height": grid.shape[0],
        "count": 1, "crs": crs, "transform": transform,
        "compress": "lzw", "nodata": 255,
    }
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(final_grid, 1)
        dst.update_tags(
            description=(
                "BCR three-way grid: "
                "0=data gap, 1=Overture only, 2=MS Buildings only, "
                "3=both datasets, 255=non-built-up (nodata)"
            ),
            reference_layer="GHS-BUILT-S R2023A E2020 threshold=10m2",
            overture_release="2025-01-22.0",
            ms_buildings_source="Microsoft Global ML Building Footprints (2023)",
        )

    size_mb = out_path.stat().st_size / 1e6
    # Coverage summary
    gap  = int((final_grid == 0).sum())
    ov   = int((final_grid == 1).sum())
    ms   = int((final_grid == 2).sum())
    both = int((final_grid == 3).sum())
    total_bu = gap + ov + ms + both
    if total_bu > 0:
        print(f"  ✓ Grid saved: {out_path.name}  ({size_mb:.1f} MB)")
        print(f"    Data gap:       {gap:>8,}  ({gap/total_bu*100:.1f}%)")
        print(f"    Overture only:  {ov:>8,}  ({ov/total_bu*100:.1f}%)")
        print(f"    MS only:        {ms:>8,}  ({ms/total_bu*100:.1f}%)")
        print(f"    Both:           {both:>8,}  ({both/total_bu*100:.1f}%)")

    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_ms_buildings_pipeline(
    country_list: list = None,
    merge_only: bool = False,
):
    """
    Full MS Buildings pipeline for all (or specified) study countries.

    Parameters
    ----------
    country_list : list of ISO3 codes, or None for all
    merge_only   : if True, skip download and only generate three-way grids
                   from already-downloaded tile files
    """
    countries = country_list or list(COUNTRIES.keys())

    # Shared index cache — download once for all countries
    cache_dir = GRID_DIR.parent / "ms_buildings_cache"
    cache_dir.mkdir(exist_ok=True)

    if not merge_only:
        index_path = download_ms_index(cache_dir)
    else:
        index_path = cache_dir / "ms_buildings_index.csv"
        if not index_path.exists():
            print("  ✗ Index not found. Run without --merge-only first.")
            return

    bcr_results = []

    for iso3 in countries:
        if iso3 not in COUNTRIES:
            print(f"  ✗ Unknown ISO3: {iso3}")
            continue

        cfg = COUNTRIES[iso3]
        print(f"\n{'='*60}")
        print(f"  MS BUILDINGS — {cfg['name']} ({iso3})")
        print(f"{'='*60}")

        # Per-country output directory
        ms_dir = cfg["folder"] / "ms_buildings"
        ms_dir.mkdir(exist_ok=True)

        # Download tiles and merge
        if not merge_only:
            ms_gpkg = download_ms_country(
                iso3       = iso3,
                bbox       = cfg["bbox"],
                index_path = index_path,
                output_dir = ms_dir,
            )
        else:
            # Find existing GPKG
            existing = list(ms_dir.glob("*.gpkg"))
            ms_gpkg  = existing[0] if existing else None
            if ms_gpkg is None:
                print(f"  ✗ No MS Buildings GPKG found for {iso3} — skipping")
                continue
            print(f"  ↳ Using existing: {ms_gpkg.name}")

        if ms_gpkg is None:
            continue

        # Compute BCR
        bcr = compute_ms_bcr(iso3, ms_gpkg)
        bcr_results.append(bcr)

        # Generate three-way completeness grid
        generate_three_way_grid(iso3, ms_gpkg)

    # Save BCR results
    if bcr_results:
        import pandas as pd
        df = pd.DataFrame(bcr_results)
        out_path = TABLE_DIR / "ms_buildings_bcr.csv"
        df.to_csv(out_path, index=False)
        print(f"\n✓ MS Buildings BCR saved: {out_path}")
        print(df[["iso3", "n_ms_buildings", "bcr_ms_national_pct"]].to_string(index=False))

    print("\n✓ MS Buildings pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Microsoft ML Building Footprints and compute BCR"
    )
    parser.add_argument(
        "--country", nargs="+", default=None,
        choices=list(COUNTRIES.keys()),
        help="ISO3 codes to process (default: all)",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip download; generate three-way grids from existing tiles",
    )
    args = parser.parse_args()
    run_ms_buildings_pipeline(
        country_list = args.country,
        merge_only   = args.merge_only,
    )