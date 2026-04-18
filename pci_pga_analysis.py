"""
=============================================================================
pci_pga_analysis.py — Place Coverage Index & Population Grid Alignment
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
Metrics:
  PCI — Place Coverage Index
        Overture places per 1,000 WorldPop residents by service category
        Computed at sub-district level; aggregated to urban agglomeration

  PGA — Population Grid Alignment
        Pearson r and MAPE between GHS-POP and WorldPop
        at sub-district level, stratified by GHSL-SMOD class
=============================================================================
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

from analysis_config import (
    COUNTRIES, PLACE_CATEGORIES, LOG_DIR, TABLE_DIR,
    PGA_MIN_POP_THRESHOLD, SMOD_CLASSES,
    URBAN_SMOD_CODES, PERIURBAN_SMOD_CODES,
)
from data_loader import (
    load_places_by_service_category, load_divisions,
    load_ghsl_raster, load_worldpop_raster,
    get_urban_extent, get_duckdb_conn,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    filename=str(LOG_DIR / "03_pci_pga.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# ZONAL STATISTICS HELPER
# ═════════════════════════════════════════════════════════════════════════════

def zonal_sum(
    raster_path_or_array,
    transform,
    crs,
    zones: gpd.GeoDataFrame,
    zone_id_col: str = "id",
    stat: str = "sum",
) -> pd.Series:
    """
    Compute zonal statistics of a raster over polygon zones.

    Uses exactextract if available (faster, more accurate for
    partial pixels), falls back to rasterstats.

    Parameters
    ----------
    raster_path_or_array : path to GeoTIFF or (array, transform, crs) tuple
    zones                : GeoDataFrame of polygons
    stat                 : "sum" or "mean"

    Returns
    -------
    pd.Series indexed by zone_id_col
    """
    # Prefer exactextract
    try:
        import exactextract
        import tempfile, os

        if isinstance(raster_path_or_array, (str, Path)):
            raster_path = str(raster_path_or_array)
        else:
            # Write array to temp file
            import rasterio
            arr, tfm, c, nd = raster_path_or_array
            tmpf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmpf.close()
            with rasterio.open(
                tmpf.name, "w", driver="GTiff", height=arr.shape[0],
                width=arr.shape[1], count=1, dtype=arr.dtype,
                crs=c, transform=tfm, nodata=nd or -9999
            ) as dst:
                dst.write(arr, 1)
            raster_path = tmpf.name

        zones_proj = zones.to_crs(str(crs)) if str(zones.crs) != str(crs) else zones
        result     = exactextract.exact_extract(
            raster_path, zones_proj, ops=[stat], output="pandas"
        )
        col_name   = f"result_{stat}"
        if col_name not in result.columns:
            col_name = result.columns[-1]
        out = pd.Series(
            result[col_name].values,
            index=zones[zone_id_col].values,
            name=stat,
        )
        if "tmpf" in dir() and os.path.exists(tmpf.name):
            os.unlink(tmpf.name)
        return out

    except ImportError:
        pass

    # Fallback: rasterstats
    try:
        from rasterstats import zonal_stats as rs_zonal
        import rasterio, tempfile, os

        if isinstance(raster_path_or_array, (str, Path)):
            raster_path = str(raster_path_or_array)
        else:
            arr, tfm, c, nd = raster_path_or_array
            tmpf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmpf.close()
            with rasterio.open(
                tmpf.name, "w", driver="GTiff", height=arr.shape[0],
                width=arr.shape[1], count=1, dtype=arr.dtype,
                crs=c, transform=tfm
            ) as dst:
                dst.write(arr, 1)
            raster_path = tmpf.name

        zones_proj = zones.to_crs(str(crs)) if str(zones.crs) != str(crs) else zones
        results    = rs_zonal(
            zones_proj, raster_path, stats=[stat], all_touched=True
        )
        values = [r[stat] if r[stat] is not None else 0.0 for r in results]
        out    = pd.Series(values, index=zones[zone_id_col].values, name=stat)

        if "tmpf" in dir() and os.path.exists(tmpf.name):
            os.unlink(tmpf.name)
        return out

    except ImportError:
        raise ImportError(
            "Neither exactextract nor rasterstats is installed.\n"
            "Install one: pip install exactextract  OR  pip install rasterstats"
        )


# ═════════════════════════════════════════════════════════════════════════════
# PCI — PLACE COVERAGE INDEX
# ═════════════════════════════════════════════════════════════════════════════

def compute_pci(
    iso3: str,
    divisions: gpd.GeoDataFrame,
    con=None,
    urban_geom=None,
) -> gpd.GeoDataFrame:
    """
    Compute Place Coverage Index at sub-district level.

    PCI = (n places in category) / (population / 1000)

    Appends PCI columns to the divisions GeoDataFrame.
    Sub-districts with WorldPop population < PGA_MIN_POP_THRESHOLD
    are assigned NaN to avoid artefactual extreme values.

    Returns
    -------
    divisions GeoDataFrame with appended columns:
      wp_pop_{iso3}          : WorldPop total population
      pci_{category}         : PCI per 1,000 residents for each category
      n_places_{category}    : raw place count per sub-district
    """
    print(f"  Computing PCI ({iso3}) ...")

    divs = divisions.to_crs("EPSG:4326").copy()

    # ── WorldPop zonal sum ────────────────────────────────────────────────────
    try:
        wp_data, wp_transform, wp_crs, wp_nodata = load_worldpop_raster(
            iso3, clip_geom=urban_geom
        )
        # Replace nodata/negative
        if wp_nodata is not None:
            wp_data = np.where(wp_data == wp_nodata, 0, wp_data)
        wp_data = np.where(wp_data < 0, 0, wp_data)

        pop_series = zonal_sum(
            (wp_data, wp_transform, wp_crs, 0),
            wp_transform, wp_crs, divs, zone_id_col="id", stat="sum"
        )
        divs["wp_pop"] = divs["id"].map(pop_series).fillna(0)
    except Exception as e:
        print(f"  ⚠ WorldPop zonal sum failed ({iso3}): {e}")
        divs["wp_pop"] = np.nan

    # ── Places spatial join ───────────────────────────────────────────────────
    if con is None:
        con = get_duckdb_conn()

    places_by_cat = load_places_by_service_category(iso3, con=con)

    for cat_name, places_gdf in places_by_cat.items():
        if len(places_gdf) == 0:
            divs[f"n_places_{cat_name}"] = 0
            divs[f"pci_{cat_name}"]      = np.nan
            continue

        places_in_div = gpd.sjoin(
            places_gdf[["id", "geometry", "service_category"]],
            divs[["id", "geometry"]].rename(columns={"id": "div_id"}),
            how="left",
            predicate="within",
        )

        count_per_div = (
            places_in_div.groupby("div_id")
            .size()
            .rename(f"n_places_{cat_name}")
        )
        divs[f"n_places_{cat_name}"] = (
            divs["id"].map(count_per_div).fillna(0).astype(int)
        )

        # PCI: per 1,000 residents; NaN where population too low
        valid_pop = divs["wp_pop"] >= PGA_MIN_POP_THRESHOLD
        divs[f"pci_{cat_name}"] = np.where(
            valid_pop,
            divs[f"n_places_{cat_name}"] / (divs["wp_pop"] / 1000),
            np.nan,
        )

    print(f"  ✓ PCI computed ({iso3})")
    return divs


# ═════════════════════════════════════════════════════════════════════════════
# PGA — POPULATION GRID ALIGNMENT
# ═════════════════════════════════════════════════════════════════════════════

def assign_smod_class(
    iso3: str,
    divisions: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each sub-district a GHSL-SMOD class based on the modal
    (most frequent) SMOD value within its boundary.

    Returns divisions with appended column: smod_modal, smod_label,
    urban_class (urban_centre / peri_urban / rural).
    """
    from rasterio.features import shapes as rio_shapes
    import rasterio

    smod_data, transform, crs, nodata = load_ghsl_raster(iso3, "GHS_SMOD")

    if nodata is not None:
        smod_data = np.where(smod_data == nodata, 0, smod_data)

    divs = divisions.to_crs(str(crs)).copy()

    # For each division, get the modal SMOD class
    # Uses zonal stats with mode = most frequent value
    try:
        from rasterstats import zonal_stats as rs_zonal
        import tempfile, os, rasterio

        tmpf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmpf.close()
        with rasterio.open(
            tmpf.name, "w", driver="GTiff", height=smod_data.shape[0],
            width=smod_data.shape[1], count=1, dtype="int16",
            crs=crs, transform=transform
        ) as dst:
            dst.write(smod_data.astype("int16"), 1)

        results = rs_zonal(divs, tmpf.name, stats=["majority"])
        os.unlink(tmpf.name)

        divs["smod_modal"] = [
            r["majority"] if r["majority"] is not None else 0
            for r in results
        ]
    except ImportError:
        # Fallback: centroid-based SMOD assignment
        from rasterio.transform import rowcol
        centroids    = divs.geometry.centroid
        rows_idx, cols_idx = rowcol(
            transform,
            centroids.x.values,
            centroids.y.values,
        )
        rows_idx = np.clip(rows_idx, 0, smod_data.shape[0]-1)
        cols_idx = np.clip(cols_idx, 0, smod_data.shape[1]-1)
        divs["smod_modal"] = smod_data[rows_idx, cols_idx]

    divs["smod_label"] = divs["smod_modal"].map(
        SMOD_CLASSES
    ).fillna("Unknown")

    divs["urban_class"] = np.select(
        [
            divs["smod_modal"].isin(URBAN_SMOD_CODES),
            divs["smod_modal"].isin(PERIURBAN_SMOD_CODES),
        ],
        ["urban_centre", "peri_urban"],
        default="rural",
    )

    return divs.to_crs("EPSG:4326")


def compute_pga(
    iso3: str,
    divisions: gpd.GeoDataFrame,
) -> dict:
    """
    Compute Population Grid Alignment between GHS-POP and WorldPop.

    Both rasters are aggregated to sub-district level using zonal sums.
    Pearson r and MAPE are computed overall and stratified by
    GHSL-SMOD urban class.

    Returns
    -------
    dict with keys: iso3, n_subdist, pearson_r, mape_all,
    mape_urban_centre, mape_peri_urban, mape_rural
    """
    print(f"  Computing PGA ({iso3}) ...")

    divs = divisions.to_crs("EPSG:4326").copy()

    # ── GHSL POP zonal sum ────────────────────────────────────────────────────
    ghsl_data, g_transform, g_crs, g_nodata = load_ghsl_raster(iso3, "GHS_POP")
    if g_nodata is not None:
        ghsl_data = np.where(ghsl_data == g_nodata, 0, ghsl_data)
    ghsl_data = np.where(ghsl_data < 0, 0, ghsl_data)

    ghsl_pop = zonal_sum(
        (ghsl_data, g_transform, g_crs, 0),
        g_transform, g_crs, divs, zone_id_col="id", stat="sum"
    )
    divs["ghsl_pop"] = divs["id"].map(ghsl_pop).fillna(0)

    # ── WorldPop zonal sum ────────────────────────────────────────────────────
    wp_data, w_transform, w_crs, w_nodata = load_worldpop_raster(iso3)
    if w_nodata is not None:
        wp_data = np.where(wp_data == w_nodata, 0, wp_data)
    wp_data = np.where(wp_data < 0, 0, wp_data)

    wp_pop = zonal_sum(
        (wp_data, w_transform, w_crs, 0),
        w_transform, w_crs, divs, zone_id_col="id", stat="sum"
    )
    divs["wp_pop"] = divs["id"].map(wp_pop).fillna(0)

    # ── Assign SMOD class ─────────────────────────────────────────────────────
    divs = assign_smod_class(iso3, divs)

    # ── Compute metrics ───────────────────────────────────────────────────────
    # Filter to sub-districts with sufficient population
    valid = (
        (divs["wp_pop"] >= PGA_MIN_POP_THRESHOLD) &
        (divs["ghsl_pop"] >= PGA_MIN_POP_THRESHOLD)
    )
    df_valid = divs[valid].copy()

    if len(df_valid) < 5:
        print(f"  ⚠ Insufficient valid sub-districts for PGA ({iso3}): {len(df_valid)}")
        return {"iso3": iso3, "n_subdist": len(df_valid), "pearson_r": np.nan}

    def mape(actual, predicted):
        """Mean Absolute Percentage Error."""
        mask = actual > 0
        return float(
            np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        )

    pearson_r, pvalue = stats.pearsonr(df_valid["wp_pop"], df_valid["ghsl_pop"])
    mape_all          = mape(df_valid["wp_pop"].values, df_valid["ghsl_pop"].values)

    # Stratified by SMOD class
    results = {
        "iso3":          iso3,
        "n_subdist":     len(df_valid),
        "pearson_r":     round(pearson_r, 4),
        "pearson_p":     round(pvalue, 6),
        "mape_all_pct":  round(mape_all, 2),
    }

    for cls in ["urban_centre", "peri_urban", "rural"]:
        subset = df_valid[df_valid["urban_class"] == cls]
        if len(subset) >= 3:
            results[f"mape_{cls}_pct"] = round(
                mape(subset["wp_pop"].values, subset["ghsl_pop"].values), 2
            )
            results[f"n_{cls}"] = len(subset)
        else:
            results[f"mape_{cls}_pct"] = np.nan
            results[f"n_{cls}"]        = len(subset)

    # Append population columns to divisions for output
    divs["pga_ghsl_pop"] = divs["ghsl_pop"]
    divs["pga_wp_pop"]   = divs["wp_pop"]
    divs["pga_diff_pct"] = np.where(
        divs["wp_pop"] > 0,
        (divs["ghsl_pop"] - divs["wp_pop"]) / divs["wp_pop"] * 100,
        np.nan
    )

    print(f"  ✓ PGA ({iso3}): r={pearson_r:.3f}, MAPE={mape_all:.1f}%")
    return results, divs


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_pci_pga_all_countries() -> tuple:
    """
    Run PCI and PGA for all seven countries.
    Returns (pci_df, pga_df) DataFrames.
    """
    print("\n" + "="*65)
    print("  PCI + PGA ANALYSIS — ALL COUNTRIES")
    print("="*65)

    pci_results  = []
    pga_results  = []
    div_all      = []
    con          = get_duckdb_conn()

    for iso3, cfg in COUNTRIES.items():
        print(f"\n── {cfg['name']} ({iso3}) ──")

        try:
            # Load divisions
            divs = load_divisions(iso3, con=con)

            # Urban extent
            urban_gdf  = get_urban_extent(iso3)
            urban_geom = urban_gdf.geometry.unary_union

            # Filter divisions to urban + peri-urban extent
            # (buffer urban extent by 20km to capture peri-urban)
            urban_buffer = (
                urban_gdf.to_crs(cfg["crs"])
                .geometry.buffer(20_000)
                .unary_union
            )
            urban_buffer_wgs = gpd.GeoSeries(
                [urban_buffer], crs=cfg["crs"]
            ).to_crs("EPSG:4326").iloc[0]

            divs_urban = divs[divs.intersects(urban_buffer_wgs)].copy()
            print(f"  {len(divs_urban)} sub-districts in urban+peri-urban zone")

            # PCI
            divs_pci = compute_pci(iso3, divs_urban, con=con,
                                   urban_geom=urban_geom)

            # PCI summary
            for cat in PLACE_CATEGORIES.keys():
                col = f"pci_{cat}"
                if col in divs_pci.columns:
                    vals = divs_pci[col].dropna()
                    pci_results.append({
                        "iso3":          iso3,
                        "country":       cfg["name"],
                        "category":      cat,
                        "n_subdist":     len(vals),
                        "pci_mean":      round(vals.mean(), 3) if len(vals) > 0 else np.nan,
                        "pci_median":    round(vals.median(), 3) if len(vals) > 0 else np.nan,
                        "pci_std":       round(vals.std(), 3) if len(vals) > 0 else np.nan,
                        "pci_min":       round(vals.min(), 3) if len(vals) > 0 else np.nan,
                        "pci_max":       round(vals.max(), 3) if len(vals) > 0 else np.nan,
                        "n_zero_pci":    int((vals == 0).sum()),
                        "pct_zero_pci":  round((vals == 0).mean() * 100, 1)
                                         if len(vals) > 0 else np.nan,
                    })

            # PGA
            pga_result = compute_pga(iso3, divs_urban)
            if isinstance(pga_result, tuple):
                pga_metrics, divs_pga = pga_result
                divs_pci = divs_pci.merge(
                    divs_pga[["id", "pga_ghsl_pop", "pga_wp_pop",
                              "pga_diff_pct", "smod_modal", "smod_label",
                              "urban_class"]],
                    on="id", how="left",
                )
            else:
                pga_metrics = pga_result

            pga_results.append(pga_metrics)
            divs_pci["iso3"] = iso3
            div_all.append(divs_pci)

        except Exception as e:
            print(f"  ✗ {iso3} failed: {e}")
            log.error(f"PCI/PGA failed for {iso3}: {e}", exc_info=True)

    # Save results
    pci_df = pd.DataFrame(pci_results)
    pga_df = pd.DataFrame(pga_results)

    pci_df.to_csv(TABLE_DIR / "pci_results.csv", index=False)
    pga_df.to_csv(TABLE_DIR / "pga_results.csv", index=False)

    # Save combined sub-district GeoPackage
    if div_all:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined = gpd.pd.concat(div_all, ignore_index=True)
            combined.to_file(
                str(TABLE_DIR.parent / "gpkg" / "benchmark_metrics_subdistrict.gpkg"),
                driver="GPKG",
            )
        print(f"  ✓ Sub-district GPKG saved")

    print(f"\n✓ PCI results: {len(pci_df)} rows")
    print(f"✓ PGA results: {len(pga_df)} countries")
    return pci_df, pga_df


if __name__ == "__main__":
    pci_df, pga_df = run_pci_pga_all_countries()
    print("\nPGA Summary:")
    print(pga_df[["iso3", "pearson_r", "mape_all_pct",
                   "mape_urban_centre_pct", "mape_peri_urban_pct"]].to_string(index=False))
