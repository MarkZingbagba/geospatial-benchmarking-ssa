"""
=============================================================================
bcr_bcc_analysis.py — Building Completeness Ratio & Attribute Completeness
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
Metrics computed:
  BCR — Building Completeness Ratio (Equation 1 in paper)
        % of GHSL built-up 100m cells containing ≥1 building centroid
  BCC — Building Category Completeness
        % of building polygons with non-null class/subtype attribute

BCR is computed at three scales:
  1. National
  2. Urban agglomeration
  3. Sub-district (appended to divisions GeoDataFrame)

Spatial join strategy:
  - GHSL built-up binary mask → rasterio pixel centres → GeoDataFrame of points
  - Overture building centroids → spatial join to GHSL pixel grid
  - Vectorised operations; no looping over rows
=============================================================================
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import xy as rio_xy
from shapely.geometry import Point, box as shapely_box
from scipy.spatial import cKDTree

from analysis_config import (
    COUNTRIES, GHSL_BUILDUP_THRESHOLD, LOG_DIR, TABLE_DIR, GRID_DIR,
)
from data_loader import (
    load_building_centroids, load_divisions,
    load_ghsl_raster, get_urban_extent,
    get_duckdb_conn,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=str(LOG_DIR / "02_bcr_bcc.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# BCR CORE FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def ghsl_buildup_to_points(
    iso3: str,
    clip_geom=None,
) -> gpd.GeoDataFrame:
    """
    Convert GHSL-BUILT-S raster to a point GeoDataFrame.

    Each point represents the centroid of a 100m GHSL cell where
    built-up surface area exceeds GHSL_BUILDUP_THRESHOLD m².

    Strategy: always load the full national raster, then spatially
    filter the resulting points to clip_geom if provided.
    This avoids CRS mismatch errors in rasterio.mask.
    """
    # Always load full raster — spatial filter applied after vectorisation
    data, transform, crs, nodata = load_ghsl_raster(
        iso3, "GHS_BUILT_S", clip_geom=None
    )

    # Apply threshold: cells with > 10 m² built-up surface
    if nodata is not None:
        data = np.where(data == nodata, 0, data)
    data = np.where(data < 0, 0, data)

    buildup_mask = data > GHSL_BUILDUP_THRESHOLD
    rows, cols   = np.where(buildup_mask)

    if len(rows) == 0:
        raise ValueError(f"No built-up cells found above threshold for {iso3}")

    # Pixel centroid coordinates
    xs, ys = rio_xy(transform, rows, cols)
    xs     = np.array(xs)
    ys     = np.array(ys)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "row":       rows,
            "col":       cols,
            "built_s":   data[rows, cols],
            "geometry":  gpd.points_from_xy(xs, ys),
        },
        crs=str(crs),
    )

    # Reproject to WGS84
    if str(crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Spatial filter by clip_geom (urban extent, in WGS84)
    # This replaces rasterio.mask clipping — avoids CRS mismatch entirely
    if clip_geom is not None:
        from shapely.geometry import shape
        mask_series = gdf.within(clip_geom)
        gdf = gdf[mask_series].copy()
        if len(gdf) == 0:
            # Try intersects as fallback (handles edge/boundary cases)
            mask_series = gdf.intersects(clip_geom)
            gdf = gdf[mask_series].copy()

    print(f"  GHSL built-up cells: {len(gdf):,} ({iso3})"
          + (" [urban extent filtered]" if clip_geom is not None else ""))
    return gdf


def compute_bcr(
    ghsl_pts: gpd.GeoDataFrame,
    building_centroids: gpd.GeoDataFrame,
    tolerance_m: float = 100.0,
) -> dict:
    """
    Compute the Building Completeness Ratio.

    Uses a KD-tree nearest-neighbour approach for speed.
    Each GHSL point is checked for the presence of at least one
    building centroid within tolerance_m metres.

    BCR = (n GHSL cells with ≥1 building centroid) / (n GHSL cells) × 100

    Parameters
    ----------
    ghsl_pts           : GHSL built-up pixel centroids (WGS84)
    building_centroids : Overture building centroids (WGS84)
    tolerance_m        : matching radius in metres (default: 100m = 1 cell width)

    Returns
    -------
    dict with keys: bcr, n_ghsl_cells, n_covered, n_gap
    """
    if len(building_centroids) == 0:
        return {"bcr": 0.0, "n_ghsl_cells": len(ghsl_pts),
                "n_covered": 0, "n_gap": len(ghsl_pts)}

    # Project both to a metric CRS for accurate distance computation
    # Use a simple equidistant projection centred on the data
    cx = ghsl_pts.geometry.x.mean()
    cy = ghsl_pts.geometry.y.mean()
    aeqd_crs = (f"+proj=aeqd +lat_0={cy:.4f} +lon_0={cx:.4f} "
                f"+datum=WGS84 +units=m")

    ghsl_proj  = ghsl_pts.to_crs(aeqd_crs)
    bldg_proj  = building_centroids.to_crs(aeqd_crs)

    # KD-tree query: for each GHSL cell, find nearest building centroid
    bldg_coords = np.column_stack([
        bldg_proj.geometry.x,
        bldg_proj.geometry.y,
    ])
    ghsl_coords = np.column_stack([
        ghsl_proj.geometry.x,
        ghsl_proj.geometry.y,
    ])

    tree       = cKDTree(bldg_coords)
    distances, _ = tree.query(ghsl_coords, k=1, workers=-1)

    covered    = distances <= tolerance_m
    n_covered  = int(covered.sum())
    n_total    = len(ghsl_pts)
    bcr        = (n_covered / n_total) * 100.0

    return {
        "bcr":          round(bcr, 2),
        "n_ghsl_cells": n_total,
        "n_covered":    n_covered,
        "n_gap":        n_total - n_covered,
        "mean_dist_m":  round(float(distances.mean()), 1),
        "pct_within_50m": round(float((distances <= 50).mean() * 100), 2),
    }


def compute_bcr_subdistrict(
    iso3: str,
    divisions: gpd.GeoDataFrame,
    building_centroids: gpd.GeoDataFrame,
    ghsl_pts: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Compute BCR for each sub-district polygon.

    Returns the divisions GeoDataFrame with appended columns:
    bcr_overture, n_ghsl_cells, n_covered, n_gap.
    """
    print(f"  Computing sub-district BCR ({iso3}, {len(divisions)} units) ...")

    # Ensure consistent CRS
    divs  = divisions.to_crs("EPSG:4326").copy()
    ghsl  = ghsl_pts.to_crs("EPSG:4326")
    bldgs = building_centroids.to_crs("EPSG:4326")

    # Spatial join GHSL points to divisions
    ghsl_in_div = gpd.sjoin(ghsl, divs[["id", "geometry"]], how="left",
                             predicate="within")
    # Spatial join buildings to divisions
    bldg_in_div = gpd.sjoin(bldgs, divs[["id", "geometry"]], how="left",
                             predicate="within")

    results = []
    for _, div_row in divs.iterrows():
        div_id   = div_row["id"]
        ghsl_sub = ghsl_in_div[ghsl_in_div["id_right"] == div_id]
        bldg_sub = bldg_in_div[bldg_in_div["id_right"] == div_id]

        n_ghsl   = len(ghsl_sub)
        if n_ghsl == 0:
            results.append({
                "id": div_id, "bcr_overture": np.nan,
                "n_ghsl_cells": 0, "n_covered": 0, "n_gap": 0,
            })
            continue

        if len(bldg_sub) == 0:
            results.append({
                "id": div_id, "bcr_overture": 0.0,
                "n_ghsl_cells": n_ghsl, "n_covered": 0, "n_gap": n_ghsl,
            })
            continue

        metrics = compute_bcr(ghsl_sub, bldg_sub)
        results.append({
            "id":            div_id,
            "bcr_overture":  metrics["bcr"],
            "n_ghsl_cells":  metrics["n_ghsl_cells"],
            "n_covered":     metrics["n_covered"],
            "n_gap":         metrics["n_gap"],
        })

    result_df = pd.DataFrame(results)
    divs = divs.merge(result_df, on="id", how="left")
    return divs


def save_bcr_grid(
    iso3: str,
    ghsl_pts: gpd.GeoDataFrame,
    building_centroids: gpd.GeoDataFrame,
    tolerance_m: float = 100.0,
):
    """
    Create and save the BCR completeness grid raster (Figure 2 data).

    Pixel values:
        0 = GHSL built-up, no footprint in either dataset (data gap)
        1 = Overture Maps only
        2 = MS Buildings only  (if available — else treated as 0)
        3 = Both datasets

    NOTE: In this dataset, only Overture Maps buildings are available.
    Extend to code 2/3 when MS Buildings are obtained.
    """
    out_path = GRID_DIR / f"{iso3}_bcr_100m.tif"
    if out_path.exists():
        print(f"  ↳ BCR grid exists: {out_path.name}")
        return str(out_path)

    # Load GHSL raster for the full country to get transform and CRS
    data, transform, crs, nodata = load_ghsl_raster(iso3, "GHS_BUILT_S")

    if nodata is not None:
        data = np.where(data == nodata, 0, data)
    buildup_mask = (data > GHSL_BUILDUP_THRESHOLD).astype(np.uint8)

    # Project buildings to GHSL CRS
    bldgs_proj = building_centroids.to_crs(str(crs))

    # Rasterise building presence: mark cells with ≥1 building centroid
    import rasterio.features
    from shapely.geometry import Point

    # Create output grid (0 = gap, 1 = Overture present)
    grid = np.zeros_like(buildup_mask, dtype=np.uint8)

    # For each building centroid, mark its cell
    from rasterio.transform import rowcol
    xs = bldgs_proj.geometry.x.values
    ys = bldgs_proj.geometry.y.values

    # Convert coordinates to pixel indices
    rows_idx, cols_idx = rowcol(transform, xs, ys)

    # Clip to raster bounds
    valid = (
        (rows_idx >= 0) & (rows_idx < grid.shape[0]) &
        (cols_idx >= 0) & (cols_idx < grid.shape[1])
    )
    rows_idx = rows_idx[valid]
    cols_idx = cols_idx[valid]

    # Mark cells: 1 = Overture present
    grid[rows_idx, cols_idx] = 1

    # Apply built-up mask: only count cells where GHSL says built-up
    # 0 in buildup_mask AND 0 in grid = non-built-up (don't count)
    # 1 in buildup_mask AND 0 in grid = DATA GAP (keep as 0 in output)
    # 1 in buildup_mask AND 1 in grid = COVERED (keep as 1)
    final_grid = np.where(buildup_mask == 1, grid, 255)  # 255 = non-built-up (nodata)

    # Save
    out_meta = {
        "driver":    "GTiff",
        "dtype":     "uint8",
        "width":     grid.shape[1],
        "height":    grid.shape[0],
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "compress":  "lzw",
        "nodata":    255,
    }

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(final_grid, 1)
        dst.update_tags(
            description="BCR Grid: 0=data gap, 1=Overture present, 255=non-built-up",
            reference="GHS-BUILT-S R2023A E2020 threshold=10m2",
        )

    print(f"  ✓ BCR grid saved: {out_path.name}")
    return str(out_path)


# ═════════════════════════════════════════════════════════════════════════════
# BCC — BUILDING CATEGORY COMPLETENESS
# ═════════════════════════════════════════════════════════════════════════════

def compute_bcc(
    iso3: str,
    con=None,
    urban_geom=None,
) -> dict:
    """
    Compute Building Category Completeness.

    BCC = % of building polygons with non-null class or subtype attribute.

    Uses DuckDB directly — does not load all geometries into memory.
    Computes national and (if urban_geom provided) urban agglomeration BCC.
    """
    from data_loader import _parquet_path

    parquet_path = _parquet_path(iso3, "buildings")

    if con is None:
        from data_loader import get_duckdb_conn
        con = get_duckdb_conn()

    # National BCC
    national_query = f"""
        SELECT
            COUNT(*)                                                AS n_total,
            COUNT(CASE WHEN class    IS NOT NULL THEN 1 END)       AS n_with_class,
            COUNT(CASE WHEN subtype  IS NOT NULL THEN 1 END)       AS n_with_subtype,
            COUNT(CASE WHEN class IS NOT NULL
                        OR subtype IS NOT NULL THEN 1 END)         AS n_with_any,
            COUNT(CASE WHEN height   IS NOT NULL THEN 1 END)       AS n_with_height,
            COUNT(CASE WHEN num_floors IS NOT NULL THEN 1 END)     AS n_with_floors
        FROM read_parquet('{parquet_path}', hive_partitioning=0)
    """

    res = con.execute(national_query).fetchone()
    n_total = res[0]

    national_bcc = {
        "iso3":                 iso3,
        "scale":                "national",
        "n_buildings":          n_total,
        "n_with_class":         res[1],
        "n_with_subtype":       res[2],
        "n_with_any_category":  res[3],
        "n_with_height":        res[4],
        "n_with_floors":        res[5],
        "bcc_class_pct":        round(res[1] / n_total * 100, 2) if n_total > 0 else 0,
        "bcc_any_pct":          round(res[3] / n_total * 100, 2) if n_total > 0 else 0,
        "bcc_height_pct":       round(res[4] / n_total * 100, 2) if n_total > 0 else 0,
    }

    # Urban agglomeration BCC (bbox filter)
    urban_bcc = None
    if urban_geom is not None:
        bounds      = urban_geom.bounds  # (minx, miny, maxx, maxy)
        urban_query = f"""
            SELECT
                COUNT(*)                                                AS n_total,
                COUNT(CASE WHEN class IS NOT NULL THEN 1 END)          AS n_with_class,
                COUNT(CASE WHEN class IS NOT NULL
                            OR subtype IS NOT NULL THEN 1 END)         AS n_with_any,
                COUNT(CASE WHEN height IS NOT NULL THEN 1 END)         AS n_with_height
            FROM read_parquet('{parquet_path}', hive_partitioning=0)
            WHERE bbox.xmin >= {bounds[0]} AND bbox.ymin >= {bounds[1]}
              AND bbox.xmax <= {bounds[2]} AND bbox.ymax <= {bounds[3]}
        """
        u_res    = con.execute(urban_query).fetchone()
        u_total  = u_res[0]
        urban_bcc = {
            "iso3":                 iso3,
            "scale":                "urban_agglomeration",
            "n_buildings":          u_total,
            "n_with_class":         u_res[1],
            "n_with_any_category":  u_res[2],
            "n_with_height":        u_res[3],
            "bcc_class_pct":        round(u_res[1] / u_total * 100, 2) if u_total > 0 else 0,
            "bcc_any_pct":          round(u_res[2] / u_total * 100, 2) if u_total > 0 else 0,
            "bcc_height_pct":       round(u_res[3] / u_total * 100, 2) if u_total > 0 else 0,
        }

    print(f"  ✓ BCC computed ({iso3}): national={national_bcc['bcc_any_pct']}%"
          + (f", urban={urban_bcc['bcc_any_pct']}%" if urban_bcc else ""))

    return {"national": national_bcc, "urban": urban_bcc}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_bcr_bcc_all_countries() -> pd.DataFrame:
    """
    Run BCR and BCC analysis for all seven study countries.
    Returns a compiled DataFrame of national-level results.
    """
    print("\n" + "="*65)
    print("  BCR + BCC ANALYSIS — ALL COUNTRIES")
    print("="*65)

    all_results = []
    con = get_duckdb_conn()

    for iso3, cfg in COUNTRIES.items():
        print(f"\n── {cfg['name']} ({iso3}) ──")

        try:
            # 1. Load urban extent
            print("  Deriving urban extent from GHSL-SMOD ...")
            urban_gdf  = get_urban_extent(iso3)
            urban_geom = urban_gdf.geometry.unary_union

            # 2. GHSL built-up points (urban extent only for urban BCR)
            print("  Building GHSL reference grid ...")
            ghsl_pts_national = ghsl_buildup_to_points(iso3)
            ghsl_pts_urban    = ghsl_buildup_to_points(iso3, clip_geom=urban_geom)

            # 3. Building centroids (national)
            print("  Loading building centroids ...")
            bldg_cent = load_building_centroids(iso3, con=con)

            # 4. BCR national
            print("  Computing national BCR ...")
            bcr_national = compute_bcr(ghsl_pts_national, bldg_cent)

            # 5. BCR urban
            print("  Computing urban agglomeration BCR ...")
            bldg_urban   = bldg_cent[bldg_cent.within(urban_geom)].copy()
            bcr_urban    = compute_bcr(ghsl_pts_urban, bldg_urban)

            # 6. BCC
            print("  Computing attribute completeness (BCC) ...")
            bcc = compute_bcc(iso3, con=con, urban_geom=urban_geom)

            # 7. BCR grid (for Figure 2)
            print("  Saving BCR grid ...")
            save_bcr_grid(iso3, ghsl_pts_national, bldg_cent)

            # Compile row
            row = {
                "iso3":                  iso3,
                "country":               cfg["name"],
                "cluster":               cfg["cluster"],
                # BCR
                "bcr_national_pct":      bcr_national["bcr"],
                "bcr_urban_pct":         bcr_urban["bcr"],
                "n_ghsl_national":       bcr_national["n_ghsl_cells"],
                "n_ghsl_urban":          bcr_urban["n_ghsl_cells"],
                "n_buildings_total":     len(bldg_cent),
                # BCC
                "bcc_national_any_pct":  bcc["national"]["bcc_any_pct"],
                "bcc_urban_any_pct":     bcc["urban"]["bcc_any_pct"] if bcc["urban"] else np.nan,
                "bcc_class_pct":         bcc["national"]["bcc_class_pct"],
                "bcc_height_pct":        bcc["national"]["bcc_height_pct"],
                # Urban area
                "urban_area_km2":        round(urban_gdf["area_km2"].sum(), 1),
            }
            all_results.append(row)
            print(f"  ✓ {iso3} complete: BCR national={bcr_national['bcr']}%, "
                  f"urban={bcr_urban['bcr']}%, BCC={bcc['national']['bcc_any_pct']}%")

        except Exception as e:
            print(f"  ✗ {iso3} failed: {e}")
            log.error(f"BCR/BCC failed for {iso3}: {e}", exc_info=True)
            all_results.append({"iso3": iso3, "country": cfg["name"],
                                 "error": str(e)})

    results_df = pd.DataFrame(all_results)

    # Save
    out_path = TABLE_DIR / "bcr_bcc_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n✓ BCR/BCC results saved: {out_path}")

    return results_df


if __name__ == "__main__":
    df = run_bcr_bcc_all_countries()
    print("\nNational BCR summary:")
    print(df[["iso3", "country", "bcr_national_pct", "bcr_urban_pct",
              "bcc_national_any_pct"]].to_string(index=False))