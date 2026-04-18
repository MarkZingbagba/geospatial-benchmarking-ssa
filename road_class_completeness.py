"""
=============================================================================
road_class_completeness.py — Road Class Completeness (RCC)
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
Computes RCC (Road Class Completeness) for Overture Maps transportation
segments relative to the total road network, at national and urban
agglomeration scales.

RCC = % of road/path segments with a non-null `class` attribute
      (e.g. primary, secondary, residential, service, path)

This metric directly supports the SDG 11.2.1 suitability score of 3
for Overture Maps: if road class attributes are consistently present,
the dataset is suitable for transport accessibility modelling.

All computation is done via DuckDB — no geometry loading required.
The roads parquet files are large (Nigeria: 971 MB, Tanzania: 1.6 GB)
so DuckDB aggregate queries are mandatory.

USAGE
─────
  python 07_road_class_completeness.py           # all countries
  python 07_road_class_completeness.py --country GHA NGA
=============================================================================
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analysis_config import COUNTRIES, LOG_DIR, TABLE_DIR
from data_loader import get_duckdb_conn, _parquet_path

logging.basicConfig(
    filename=str(LOG_DIR / "07_rcc.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def compute_rcc_national(iso3: str, con=None) -> dict:
    """
    Compute national-level Road Class Completeness for Overture Maps.

    RCC_national = % of all road segments with non-null `class` attribute

    Uses DuckDB directly on the roads parquet file.
    Also reports sub-type breakdown (primary, secondary, residential, etc.)
    for qualitative richness assessment.
    """
    if con is None:
        con = get_duckdb_conn()

    try:
        road_path = _parquet_path(iso3, "roads")
    except FileNotFoundError:
        print(f"  ✗ {iso3}: roads.parquet not found — skipping")
        return {"iso3": iso3, "error": "roads.parquet not found"}

    print(f"  Computing RCC ({iso3}) — {road_path.stat().st_size/1e6:.0f} MB ...")

    # ── National RCC ─────────────────────────────────────────────────────────
    national_query = f"""
        SELECT
            COUNT(*)                                            AS n_total,
            COUNT(CASE WHEN class IS NOT NULL THEN 1 END)      AS n_with_class,
            COUNT(CASE WHEN subtype IS NOT NULL THEN 1 END)    AS n_with_subtype,
            COUNT(CASE WHEN class IS NOT NULL
                        OR subtype IS NOT NULL THEN 1 END)     AS n_with_any
        FROM read_parquet('{road_path}', hive_partitioning=0)
    """
    res = con.execute(national_query).fetchone()
    n_total = res[0]
    if n_total == 0:
        return {"iso3": iso3, "error": "no road segments found"}

    rcc_class  = round(res[1] / n_total * 100, 2)
    rcc_any    = round(res[3] / n_total * 100, 2)

    # ── Class value distribution ──────────────────────────────────────────────
    class_dist_query = f"""
        SELECT
            class,
            COUNT(*) AS n
        FROM read_parquet('{road_path}', hive_partitioning=0)
        WHERE class IS NOT NULL
        GROUP BY class
        ORDER BY n DESC
        LIMIT 15
    """
    class_dist = con.execute(class_dist_query).df()

    # Format top classes as a readable string
    top_classes = "; ".join(
        f"{row['class']}={row['n']:,}"
        for _, row in class_dist.head(6).iterrows()
    )

    result = {
        "iso3":              iso3,
        "country":           COUNTRIES[iso3]["name"],
        "n_road_segments":   n_total,
        "n_with_class":      res[1],
        "n_with_subtype":    res[2],
        "n_with_any":        res[3],
        "rcc_class_pct":     rcc_class,
        "rcc_any_pct":       rcc_any,
        "top_classes":       top_classes,
    }

    print(f"  ✓ RCC ({iso3}): {rcc_class:.1f}% have class attribute  "
          f"({n_total:,} total segments)")
    log.info(f"[OK] RCC {iso3}: {rcc_class:.1f}%  n={n_total:,}")
    return result


def compute_rcc_urban(iso3: str, con=None) -> dict:
    """
    Compute urban agglomeration RCC using bbox filter derived from
    the GHSL-SMOD urban extent.

    This avoids loading road geometries — uses bbox.xmin/ymin/xmax/ymax
    fields for a fast spatial pre-filter.
    """
    if con is None:
        con = get_duckdb_conn()

    from data_loader import get_urban_extent
    try:
        urban_gdf  = get_urban_extent(iso3)
        urban_geom = urban_gdf.geometry.unary_union
        bounds     = urban_geom.bounds  # (minx, miny, maxx, maxy)
    except Exception as e:
        print(f"  ⚠ {iso3}: could not derive urban extent: {e}")
        return {}

    try:
        road_path = _parquet_path(iso3, "roads")
    except FileNotFoundError:
        return {}

    urban_query = f"""
        SELECT
            COUNT(*)                                            AS n_total,
            COUNT(CASE WHEN class IS NOT NULL THEN 1 END)      AS n_with_class,
            COUNT(CASE WHEN class IS NOT NULL
                        OR subtype IS NOT NULL THEN 1 END)     AS n_with_any
        FROM read_parquet('{road_path}', hive_partitioning=0)
        WHERE bbox.xmin >= {bounds[0]} AND bbox.ymin >= {bounds[1]}
          AND bbox.xmax <= {bounds[2]} AND bbox.ymax <= {bounds[3]}
    """
    res = con.execute(urban_query).fetchone()
    n_total = res[0]
    if n_total == 0:
        return {"iso3": iso3, "rcc_urban_pct": None}

    return {
        "iso3":                   iso3,
        "n_road_segments_urban":  n_total,
        "n_with_class_urban":     res[1],
        "rcc_urban_pct":          round(res[1] / n_total * 100, 2),
    }


def run_rcc_all_countries(country_list=None) -> pd.DataFrame:
    """
    Run RCC for all study countries and save results.
    """
    print("\n" + "="*60)
    print("  ROAD CLASS COMPLETENESS (RCC) — ALL COUNTRIES")
    print("="*60)

    countries = country_list or list(COUNTRIES.keys())
    con       = get_duckdb_conn()
    results   = []

    for iso3 in countries:
        if iso3 not in COUNTRIES:
            continue
        print(f"\n── {COUNTRIES[iso3]['name']} ({iso3}) ──")

        national = compute_rcc_national(iso3, con)
        urban    = compute_rcc_urban(iso3, con)

        if "error" not in national:
            row = {**national, **urban}
            results.append(row)
        else:
            results.append(national)

    df = pd.DataFrame(results)

    # Save
    out_path = TABLE_DIR / "rcc_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ RCC results saved: {out_path}")

    # Summary print
    print("\nRoad Class Completeness Summary:")
    print("-"*60)
    cols = ["iso3", "country", "n_road_segments", "rcc_class_pct", "rcc_urban_pct"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Road Class Completeness (RCC) for Overture Maps"
    )
    parser.add_argument(
        "--country", nargs="+", default=None,
        choices=list(COUNTRIES.keys()),
    )
    args = parser.parse_args()
    run_rcc_all_countries(args.country)
