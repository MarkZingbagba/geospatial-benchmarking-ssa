"""
=============================================================================
figures.py — Publication-Quality Figures for Scientific Data
Paper: Benchmarking Open Geospatial Data Stacks
=============================================================================
Produces all five figures required for the paper:

  Figure 1 — Study area map with urban agglomeration extents
  Figure 2 — BCR completeness grids (7 cities, panel layout)
  Figure 3 — GHS-POP vs WorldPop scatter plots (7 countries)
  Figure 4 — SDG 11 suitability matrix heatmap
  Figure 5 — Pipeline workflow diagram

All figures:
  - 300 dpi minimum (Scientific Data requirement)
  - Colourblind-accessible palettes (viridis, ColorBrewer)
  - Consistent font: Helvetica Neue / Arial / Liberation Sans
  - Output as PDF (for LaTeX) and PNG (for review/inspection)
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

from analysis_config import (
    COUNTRIES, FIG_DIR, FIG_DPI, TABLE_DIR, GRID_DIR,
    SUITABILITY_COLOURS, SUITABILITY_LABELS, COLOURS,
)

# SDG 11 indicator short labels for figure axes
SDG11_INDICATORS = [
    "11.1.1_informal_settlements",
    "11.2.1_transport_access",
    "11.3.1_land_efficiency",
    "11.6.1_waste_management",
    "11.6.2_air_quality_proxy",
    "11.7.1_open_space",
]

# ── Matplotlib style settings ─────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica Neue", "Liberation Sans", "DejaVu Sans"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,        # screen; set to 300 for final save
    "savefig.dpi":        FIG_DPI,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# Country order for consistent panel layout
COUNTRY_ORDER = ["GHA", "CIV", "SEN", "NGA", "CMR", "KEN", "TZA"]
COUNTRY_NAMES = {k: v["name"] for k, v in COUNTRIES.items()}


def _save_fig(fig, name: str, tight: bool = True):
    """Save figure as both PDF and PNG."""
    if tight:
        fig.tight_layout()
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
    fig.savefig(str(png_path), format="png", dpi=FIG_DPI, bbox_inches="tight")
    print(f"  ✓ Figure saved: {pdf_path.name}  |  {png_path.name}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Study Area Map
# ═════════════════════════════════════════════════════════════════════════════

def plot_study_area():
    """
    Figure 1: Study area overview showing the 7 countries and their
    primary urban agglomerations on an SSA basemap.

    Requires geopandas with natural earth data.
    """
    try:
        import geopandas as gpd

        # Natural Earth admin0 boundaries
        # geopandas.datasets was removed in GeoPandas 1.0.
        # Use geodatasets (the official replacement) or fall back to URL download.
        world = None

        # Method 1: geodatasets package (pip install geodatasets)
        try:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path("naturalearth lowres"))
        except Exception:
            pass

        # Method 2: download directly from Natural Earth (requires internet)
        if world is None:
            try:
                import requests, tempfile, zipfile, os
                url = (
                    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
                    "ne_110m_admin_0_countries.zip"
                )
                cache_path = FIG_DIR / "ne_110m_countries.gpkg"
                if cache_path.exists():
                    world = gpd.read_file(str(cache_path))
                else:
                    print("  Downloading Natural Earth boundaries ...")
                    resp = requests.get(url, timeout=60)
                    resp.raise_for_status()
                    with tempfile.TemporaryDirectory() as tmp:
                        zp = os.path.join(tmp, "ne.zip")
                        with open(zp, "wb") as zf:
                            zf.write(resp.content)
                        with zipfile.ZipFile(zp) as z:
                            z.extractall(tmp)
                        shp = [f for f in os.listdir(tmp) if f.endswith(".shp")]
                        if shp:
                            world = gpd.read_file(os.path.join(tmp, shp[0]))
                            world.to_file(str(cache_path), driver="GPKG")
                            print(f"  ✓ Natural Earth data cached: {cache_path.name}")
            except Exception as e2:
                print(f"  ⚠ Download failed: {e2}")

        if world is None:
            raise RuntimeError(
                "Could not load Natural Earth data.\n"
                "Fix: pip install geodatasets\n"
                "Then re-run: python run_analysis.py --step figures"
            )

    except Exception as e:
        print(f"  ⚠ Figure 1 skipped: {e}")
        print("    Fix: pip install geodatasets")
        return

    study_isos = list(COUNTRIES.keys())

    # Detect ISO3 column — name varies by Natural Earth source/version
    iso3_col = next(
        (c for c in ["iso_a3", "ISO_A3", "ADM0_A3", "SOV_A3", "ISO3"]
         if c in world.columns),
        None
    )
    if iso3_col is None:
        print(f"  ⚠ Figure 1 skipped: ISO3 column not found.")
        print(f"    Available columns: {list(world.columns[:10])}")
        return

    study      = world[world[iso3_col].isin(study_isos)]
    ssa        = world[
        (world.geometry.centroid.y < 20) &
        (world.geometry.centroid.y > -35) &
        (world.geometry.centroid.x > -20) &
        (world.geometry.centroid.x < 52)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # Left panel: SSA overview
    ax0 = axes[0]
    ssa.plot(ax=ax0, color="#f0f0f0", edgecolor="#999999", linewidth=0.4)
    study.plot(ax=ax0, color="#fee08b", edgecolor="#d73027", linewidth=0.8)

    # Annotate capitals
    for iso3, cfg in COUNTRIES.items():
        sub     = study[study[iso3_col] == iso3]
        if sub.empty:
            continue
        cx      = sub.geometry.centroid.x.values[0]
        cy      = sub.geometry.centroid.y.values[0]
        ax0.annotate(
            iso3, (cx, cy),
            fontsize=7, ha="center", va="center",
            fontweight="bold", color="#333333",
        )

    ax0.set_xlim(-20, 52)
    ax0.set_ylim(-35, 22)
    ax0.set_xlabel("Longitude (°E/W)", fontsize=8)
    ax0.set_ylabel("Latitude (°N/S)", fontsize=8)
    ax0.set_title("Study countries in Sub-Saharan Africa", fontsize=10,
                  fontweight="bold")

    # Add legend
    patches = [
        mpatches.Patch(color="#fee08b", ec="#d73027", label="Study countries (n=7)"),
        mpatches.Patch(color="#f0f0f0", ec="#999999", label="Other SSA countries"),
    ]
    ax0.legend(handles=patches, loc="lower left", fontsize=7)

    # Right panel: cluster breakdown table
    ax1 = axes[1]
    ax1.axis("off")

    table_data = [
        ["ISO3", "Country", "Cluster", "Capital"],
    ]
    for iso3, cfg in COUNTRIES.items():
        table_data.append([
            iso3, cfg["name"], cfg["cluster"].split("(")[0].strip(), cfg["capital"]
        ])

    tbl = ax1.table(
        cellText   = table_data[1:],
        colLabels  = table_data[0],
        loc        = "center",
        cellLoc    = "left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    ax1.set_title("Study country summary", fontsize=10, fontweight="bold")

    _save_fig(fig, "fig1_study_area_map")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — BCR Results Bar Chart + Completeness Comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_bcr_results(bcr_df: pd.DataFrame, ms_bcr_df: pd.DataFrame = None):
    """
    Figure 2: Building Completeness Ratio (BCR) and Attribute Completeness (BCC).

    Layout: three panels side by side.
      Panel A: National BCR — Overture Maps vs MS Buildings (grouped bars)
      Panel B: Urban agglomeration BCR — Overture Maps only
      Panel C: Building Category Completeness (BCC)

    The key finding — Overture outperforms MS Buildings nationally in all
    seven countries — is shown by direct side-by-side comparison in Panel A.
    This challenges prior literature and is the central novel empirical result.
    """
    if bcr_df.empty:
        print("  ⚠ Figure 2 skipped: BCR results not available.")
        return

    has_ms = (ms_bcr_df is not None and not ms_bcr_df.empty
              and "bcr_ms_national_pct" in ms_bcr_df.columns)

    df = bcr_df[bcr_df["iso3"].isin(COUNTRY_ORDER)].copy()
    df["country_short"] = df["iso3"].map({
        k: v["name"].replace("Côte d'Ivoire", "C.d'Ivoire")
        for k, v in COUNTRIES.items()
    })

    if has_ms:
        df = df.merge(
            ms_bcr_df[["iso3", "bcr_ms_national_pct", "n_ms_buildings"]],
            on="iso3", how="left"
        )

    # Sort by Overture national BCR ascending
    df = df.sort_values("bcr_national_pct", ascending=True).reset_index(drop=True)

    # Three panels if MS data available, two otherwise
    ncols  = 3 if has_ms else 2
    figw   = 16 if has_ms else 12
    fig, axes = plt.subplots(1, ncols, figsize=(figw, 6))
    if ncols == 2:
        axes = list(axes) + [None]

    y      = np.arange(len(df))
    width  = 0.35

    # ── Panel A: National BCR comparison ─────────────────────────────────────
    ax = axes[0]
    if has_ms:
        # Side-by-side: Overture (top) vs MS Buildings (bottom) per country
        bars_ov = ax.barh(y + width/2, df["bcr_national_pct"], width,
                          label="Overture Maps", color="#2166ac", alpha=0.88)
        bars_ms = ax.barh(y - width/2, df["bcr_ms_national_pct"], width,
                          label="MS Buildings", color="#b2182b", alpha=0.75)

        for bar in bars_ov:
            w = bar.get_width()
            if not np.isnan(w):
                ax.text(w + 0.8, bar.get_y() + bar.get_height()/2,
                        f"{w:.1f}%", va="center", fontsize=7, color="#2166ac")
        for bar in bars_ms:
            w = bar.get_width()
            if not np.isnan(w):
                ax.text(w + 0.8, bar.get_y() + bar.get_height()/2,
                        f"{w:.1f}%", va="center", fontsize=7, color="#b2182b")

        ax.set_title("National BCR\n(Overture Maps vs MS Buildings)",
                     fontweight="bold", fontsize=9)
        ax.legend(loc="lower right", fontsize=8)

        # Annotate the mean gap — placed lower-LEFT to avoid legend overlap
        if "bcr_ms_national_pct" in df.columns:
            mean_gap = (df["bcr_national_pct"] - df["bcr_ms_national_pct"]).mean()
            ax.text(0.03, 0.03,
                    f"Mean gap:\nOverture +{mean_gap:.1f} pp\nabove MS Buildings",
                    transform=ax.transAxes, fontsize=7.5,
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#e8f4f8",
                              edgecolor="#2166ac", linewidth=1.2, alpha=0.92))
    else:
        # Without MS data — single Overture bar
        bars_nat = ax.barh(y, df["bcr_national_pct"], color="#2166ac", alpha=0.88,
                           label="National BCR")
        for bar in bars_nat:
            w = bar.get_width()
            if not np.isnan(w):
                ax.text(w + 0.8, bar.get_y() + bar.get_height()/2,
                        f"{w:.1f}%", va="center", fontsize=7, color="#2166ac")
        ax.set_title("National BCR (Overture Maps)", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df["country_short"])
    ax.set_xlabel("Building Completeness Ratio (%)\n"
                  "[vs GHS-BUILT-S R2023A reference]", fontsize=8)
    ax.axvline(x=50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(0, 110)
    ax.tick_params(axis="y", labelsize=8)

    # ── Panel B: Urban BCR (Overture only) ───────────────────────────────────
    ax2 = axes[1]
    bars_urb = ax2.barh(y, df["bcr_urban_pct"], color="#fc8d59", alpha=0.88)
    for bar in bars_urb:
        w = bar.get_width()
        if not np.isnan(w):
            ax2.text(w + 0.8, bar.get_y() + bar.get_height()/2,
                     f"{w:.1f}%", va="center", fontsize=7, color="#d94701")
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(df))   # no repeated labels
    ax2.tick_params(axis="y", length=5, color="#cccccc")  # keep ticks for row tracking
    ax2.set_xlabel("Urban Agglomeration BCR (%)\n"
                   "[Overture Maps, GHSL-SMOD urban extent]", fontsize=8)
    ax2.set_title("Urban Agglomeration BCR\n(Overture Maps)",
                  fontweight="bold", fontsize=9)
    ax2.set_xlim(0, 110)

    # ── Panel C: BCC (if three panels) ───────────────────────────────────────
    if axes[2] is not None:
        ax3 = axes[2]
        if "bcc_national_any_pct" in df.columns:
            bars_bcc = ax3.barh(y, df["bcc_national_any_pct"],
                                color="#4dac26", alpha=0.85)
            for bar in bars_bcc:
                w = bar.get_width()
                if not np.isnan(w):
                    ax3.text(w + 0.05, bar.get_y() + bar.get_height()/2,
                             f"{w:.2f}%", va="center", fontsize=7,
                             color="#1a7523")
            ax3.set_yticks(y)
            ax3.set_yticklabels([""] * len(df))
            ax3.tick_params(axis="y", length=5, color="#cccccc")  # row tracking
            ax3.set_xlabel("Building Category Completeness (%)\n"
                           "[Proportion with non-null class or subtype]", fontsize=8)
            ax3.set_title("Attribute Completeness (BCC)\n(Overture Maps)",
                          fontweight="bold", fontsize=9)
            # Panel C x-axis: cap at 35% above max value so near-zero bars
            # have visible width (floor at 2 so smallest bars show as a sliver)
            max_bcc  = df["bcc_national_any_pct"].max()
            mean_bcc = df["bcc_national_any_pct"].mean()
            ax3.set_xlim(0, max(max_bcc * 1.35, 2))

            # Mean reference line — anchors the near-zero values
            ax3.axvline(x=mean_bcc, color="#999999",
                        linestyle=":", linewidth=1.0, alpha=0.7)
            ax3.text(mean_bcc + 0.05, len(df) - 0.1,
                     f"Mean: {mean_bcc:.1f}%",
                     fontsize=6.5, color="#777777", va="top")

            ax3.text(0.97, 0.02,
                     f"Note: BCC near-zero\n(mean: {mean_bcc:.1f}%)\n"
                     "indicates geometry-only\ncoverage in 2025 release",
                     transform=ax3.transAxes, fontsize=7, color="#555",
                     va="bottom", ha="right",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9f0",
                               edgecolor="#999", alpha=0.85))

    fig.suptitle(
        "Overture Maps vs Microsoft ML Buildings — Building Completeness Ratio (BCR)\n"
        "Reference: GHS-BUILT-S R2023A (epoch 2020), threshold >10 m² per 100 m cell",
        fontsize=10, fontweight="bold",
    )
    _save_fig(fig, "fig2_bcr_bcc_results")


def plot_pga_scatterplots(pga_df: pd.DataFrame, subdistrict_gpkg: str = None):
    """
    Figure 3: GHS-POP vs WorldPop scatter plots for all seven countries.

    Key design decisions for Scientific Data reviewers:
    - CIV (MAPE 143.8%) and CMR (MAPE 252.5%) use LOG-SCALE axes because
      a small number of very large sub-districts (rural communes) have
      population totals 10-100x larger than the majority. On linear scale,
      95% of points cluster invisibly in the lower-left corner. Log scale
      reveals the full distribution and makes the systematic overestimation
      by GHS-POP visible across ALL population ranges, not just outliers.
    - Linear-scale countries (GHA, SEN, NGA, KEN, TZA) use consistent
      k-formatted tick labels.
    - Each panel carries a scale indicator badge ("log" or "linear") to
      prevent misreading across the 3x3 panel layout.
    - Regression line for log-scale panels is fitted on log-transformed
      data and plotted in log space (correct), not on raw values (incorrect).
    """
    from scipy import stats as sp_stats

    if pga_df.empty:
        print("  ⚠ Figure 3 skipped: PGA results not available.")
        return

    # Countries requiring log-scale treatment
    # Criterion: MAPE > 100% indicating extreme scale dispersion
    LOG_SCALE_COUNTRIES = {"CIV", "CMR"}

    # Load sub-district data if available
    gdf_all = None
    if subdistrict_gpkg and Path(subdistrict_gpkg).exists():
        import geopandas as gpd
        try:
            gdf_all = gpd.read_file(subdistrict_gpkg)
        except Exception:
            pass

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    axes_flat = axes.flatten()

    colour_map = {
        "urban_centre": COLOURS["urban_centre"],
        "peri_urban":   COLOURS["periurban"],
        "rural":        COLOURS["rural"],
    }
    label_map = {
        "urban_centre": "Urban centre",
        "peri_urban":   "Peri-urban",
        "rural":        "Rural",
    }

    def fmt_k(x, _):
        """Format axis tick as '150k', '1.2M', etc."""
        if x <= 0:
            return ""
        if x >= 1e6:
            return f"{x/1e6:.1f}M"
        if x >= 1e3:
            return f"{x/1e3:.0f}k"
        return f"{x:.0f}"

    for idx, iso3 in enumerate(COUNTRY_ORDER):
        ax  = axes_flat[idx]
        cfg = COUNTRIES[iso3]
        use_log = iso3 in LOG_SCALE_COUNTRIES

        # Load sub-district data for this country
        if gdf_all is not None and "iso3" in gdf_all.columns:
            sub = gdf_all[
                (gdf_all["iso3"] == iso3) &
                gdf_all["pga_wp_pop"].notna() &
                gdf_all["pga_ghsl_pop"].notna() &
                (gdf_all["pga_wp_pop"] > 0) &
                (gdf_all["pga_ghsl_pop"] > 0)
            ].copy()
        else:
            sub = None

        if sub is not None and len(sub) > 0:
            # ── Set axis scale BEFORE plotting ───────────────────────────────
            if use_log:
                ax.set_xscale("log")
                ax.set_yscale("log")

            # ── Plot points by SMOD class ─────────────────────────────────────
            for cls, colour in colour_map.items():
                mask = sub.get("urban_class", pd.Series(dtype=str)) == cls
                if mask.any():
                    ax.scatter(
                        sub.loc[mask, "pga_wp_pop"],
                        sub.loc[mask, "pga_ghsl_pop"],
                        c=colour, alpha=0.6, s=15,
                        label=label_map[cls], linewidths=0,
                    )

            # ── 1:1 reference line ────────────────────────────────────────────
            if use_log:
                # Log-space reference line spanning full data range
                lo = max(1, min(sub["pga_wp_pop"].min(),
                                sub["pga_ghsl_pop"].min()))
                hi = max(sub["pga_wp_pop"].max(), sub["pga_ghsl_pop"].max())
                ref_x = np.logspace(np.log10(lo), np.log10(hi), 200)
                ax.plot(ref_x, ref_x, "k--", linewidth=0.9, alpha=0.5,
                        zorder=1)
            else:
                hi = max(sub["pga_wp_pop"].max(), sub["pga_ghsl_pop"].max())
                ax.plot([0, hi], [0, hi], "k--", linewidth=0.9, alpha=0.5,
                        zorder=1)

            # ── OLS regression line ───────────────────────────────────────────
            valid = sub[["pga_wp_pop", "pga_ghsl_pop"]].dropna()
            valid = valid[(valid["pga_wp_pop"] > 0) & (valid["pga_ghsl_pop"] > 0)]
            if len(valid) > 3:
                if use_log:
                    # Fit in log space (geometrically meaningful for log axes)
                    lx = np.log10(valid["pga_wp_pop"])
                    ly = np.log10(valid["pga_ghsl_pop"])
                    slope, intercept, *_ = sp_stats.linregress(lx, ly)
                    xs_log = np.linspace(lx.min(), lx.max(), 200)
                    ax.plot(10**xs_log,
                            10**(slope * xs_log + intercept),
                            "r-", linewidth=1.2, alpha=0.8, zorder=2)
                else:
                    slope, intercept, *_ = sp_stats.linregress(
                        valid["pga_wp_pop"], valid["pga_ghsl_pop"]
                    )
                    xs = np.linspace(0, valid["pga_wp_pop"].max(), 200)
                    ax.plot(xs, slope * xs + intercept,
                            "r-", linewidth=1.2, alpha=0.8, zorder=2)

            # ── Axis tick formatting ──────────────────────────────────────────
            if use_log:
                # Log axes: use concise magnitude labels on major ticks only
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            else:
                ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))

        # ── Annotation box: r, MAPE, scale ───────────────────────────────────
        pga_row = pga_df[pga_df["iso3"] == iso3]
        if not pga_row.empty:
            r_val    = pga_row["pearson_r"].values[0]
            mape_val = pga_row["mape_all_pct"].values[0]
            scale_tag = "log scale" if use_log else "linear scale"
            annot = f"r = {r_val:.3f}\nMAPE = {mape_val:.1f}%\n[{scale_tag}]"
            # Log-scale panels: place annotation in upper-left in log space
            ax.text(0.05, 0.95, annot,
                    transform=ax.transAxes,
                    fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white",
                              edgecolor="#d73027" if use_log else "grey",
                              linewidth=1.2 if use_log else 0.8,
                              alpha=0.90))

        # ── Log-scale visual warning badge ───────────────────────────────────
        if use_log:
            ax.text(0.99, 0.01, "⚠ log axes",
                    transform=ax.transAxes,
                    fontsize=7, ha="right", va="bottom",
                    color="#d73027", fontstyle="italic")

        ax.set_title(cfg["name"], fontweight="bold", fontsize=9)
        ax.set_xlabel("WorldPop 2020 (persons)", fontsize=7)
        ax.set_ylabel("GHS-POP 2020 (persons)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)

    # ── Legend panel (bottom-right) ───────────────────────────────────────────
    if len(COUNTRY_ORDER) < len(axes_flat):
        for extra_ax in axes_flat[len(COUNTRY_ORDER):]:
            extra_ax.axis("off")

        legend_ax = axes_flat[-1]
        legend_ax.axis("off")

        handles = [
            mpatches.Patch(color=colour_map[c], label=label_map[c])
            for c in colour_map
        ]
        handles += [
            plt.Line2D([0], [0], color="k", linestyle="--",
                       label="1:1 reference line"),
            plt.Line2D([0], [0], color="r", linestyle="-",
                       label="OLS regression"),
        ]
        # Log-scale explanation
        handles += [
            mpatches.Patch(facecolor="none",
                           edgecolor="#d73027", linewidth=1.5,
                           label="Red border = log axes\n(MAPE > 100%)"),
        ]
        legend_ax.legend(handles=handles, loc="center", fontsize=8.5,
                         title="GHSL-SMOD Class", title_fontsize=9,
                         framealpha=0.9)

        # Explanatory note about log scale choice
        legend_ax.text(0.5, 0.12,
            "Note: Côte d'Ivoire and Cameroon\n"
            "plotted on log–log axes owing to\n"
            "large rural commune populations\n"
            "(MAPE > 100%). Regression fitted\n"
            "in log space. Both products agree\n"
            "on relative distribution (r > 0.97)\n"
            "but diverge in absolute counts,\n"
            "reflecting census uncertainty.",
            transform=legend_ax.transAxes,
            ha="center", va="bottom", fontsize=7.5,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#fff9f0",
                      edgecolor="#d73027",
                      linewidth=0.8,
                      alpha=0.9))

    fig.suptitle(
        "Sub-District Population Comparison: GHS-POP vs WorldPop (2020)\n"
        "Points = sub-districts; colour = GHSL Degree of Urbanisation class. "
        "Côte d'Ivoire and Cameroon: log–log axes (see note).",
        fontsize=10, y=1.01,
    )
    _save_fig(fig, "fig3_population_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — SDG 11 Suitability Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def plot_suitability_heatmap(suitability_df: pd.DataFrame):
    """
    Figure 4: SDG 11 suitability matrix as an annotated heatmap.
    Colourblind-accessible: red–yellow–green diverging palette.
    """
    if suitability_df is None or suitability_df.empty:
        print("  ⚠ Figure 4 skipped: suitability matrix not available.")
        return

    # Drop fusion row and any non-numeric rows for heatmap
    numeric_df = suitability_df.drop(
        index=["Best_Fusion"] if "Best_Fusion" in suitability_df.index else [],
        errors="ignore",
    )

    # Extract fusion row separately
    fusion_row = suitability_df.loc["Best_Fusion"] \
        if "Best_Fusion" in suitability_df.index else None

    # Build numeric matrix
    try:
        matrix = numeric_df.astype(float)
    except ValueError:
        print("  ⚠ Figure 4: Could not convert suitability matrix to float.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))

    # Custom colourmap: red (0) → yellow (1) → light green (2) → dark green (3)
    cmap   = ListedColormap(["#d73027", "#fee08b", "#a6d96a", "#1a9641"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm   = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix.values, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val == 0 else "black"
                ax.text(j, i, f"{int(val)}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    # Axis labels
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [col.replace("_", " ").replace("11.", "SDG 11.") for col in matrix.columns],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(
        [idx.replace("_", " ") for idx in matrix.index],
        fontsize=9,
    )
    ax.set_title(
        "SDG 11 Data Suitability Matrix\n"
        "Score: 0 = Not suitable | 1 = Partial | 2 = Suitable | 3 = Fully suitable",
        fontsize=10, fontweight="bold",
    )

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], pad=0.01)
    cbar.ax.set_yticklabels(
        [SUITABILITY_LABELS[s] for s in [0, 1, 2, 3]], fontsize=7
    )

    # Add fusion row as text below heatmap
    if fusion_row is not None:
        fusion_text = "Best fusion: " + " | ".join(
            f"{col.split('_')[0]}: {val}"
            for col, val in fusion_row.items()
            if not pd.isna(val)
        )
        fig.text(0.01, -0.05, fusion_text[:120], fontsize=6.5,
                 color="grey", wrap=True)

    _save_fig(fig, "fig4_suitability_heatmap")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Pipeline Workflow Diagram
# ═════════════════════════════════════════════════════════════════════════════

def plot_pipeline_workflow():
    """
    Figure 5: Pipeline workflow as a 2x2 grid — fits within page width.

    Layout:
      [Stage 1: Data Acquisition] ──► [Stage 2: Harmonisation]
                  │                              │
                  ▼                              ▼
      [Stage 4: Synthesis]        ◄── [Stage 3: Metric Computation]

    This Z-flow reads naturally and avoids the overflow of a 4-column
    horizontal layout.
    """
    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    dataset_colour = "#d08b4e"   # warm tan — input datasets
    process_colour = "#dce9f5"   # light blue — processing steps
    output_colour  = "#7fb899"   # sage green — outputs
    border_colour  = "#2c7bb6"
    arrow_colour   = "#444444"

    # ── Stage positions (x, y of bottom-left corner) ──────────────────────
    # 2×2 grid with generous margins
    BOX_W, BOX_H = 4.8, 3.0
    pad          = 0.25   # inner padding

    positions = {
        "S1": (0.6,  4.4),   # top-left
        "S2": (6.6,  4.4),   # top-right
        "S3": (6.6,  0.7),   # bottom-right
        "S4": (0.6,  0.7),   # bottom-left
    }

    stages = {
        "S1": {
            "title":   "Stage 1 — Data Acquisition",
            "items":   [
                ("Overture Maps CLI",  dataset_colour),
                ("GHSL (JRC FTP)",     dataset_colour),
                ("WorldPop Hub API",   dataset_colour),
            ],
        },
        "S2": {
            "title":   "Stage 2 — Harmonisation",
            "items":   [
                ("Reproject → local UTM",   process_colour),
                ("Clip to urban extent",    process_colour),
                ("Convert → GeoPackage",    process_colour),
            ],
        },
        "S3": {
            "title":   "Stage 3 — Metric Computation",
            "items":   [
                ("BCR — Building Completeness (Eq. 1)",  process_colour),
                ("BCC — Attribute Completeness",         process_colour),
                ("PCI — Place Coverage Index",           process_colour),
                ("PGA — Population Alignment (Eq. 2)",   process_colour),
            ],
        },
        "S4": {
            "title":   "Stage 4 — Synthesis & Outputs",
            "items":   [
                ("Benchmark tables (national + urban)", output_colour),
                ("BCR completeness grids (GeoTIFF)",    output_colour),
                ("SDG 11 suitability matrix",           output_colour),
            ],
        },
    }

    stage_order  = ["S1", "S2", "S3", "S4"]
    stage_colours = {
        "S1": "#4a7fb5",
        "S2": "#6199c6",
        "S3": "#7ab3d4",
        "S4": "#c8dff0",
    }

    item_h = 0.48

    for sid in stage_order:
        x, y   = positions[sid]
        stage  = stages[sid]
        col    = stage_colours[sid]
        items  = stage["items"]

        # Outer box
        outer = mpatches.FancyBboxPatch(
            (x, y), BOX_W, BOX_H,
            boxstyle="round,pad=0.12",
            facecolor=col, edgecolor=border_colour,
            linewidth=1.5, zorder=1,
        )
        ax.add_patch(outer)

        # Stage title inside box at top
        ax.text(x + BOX_W/2, y + BOX_H - 0.22,
                stage["title"],
                ha="center", va="top",
                fontsize=9.5, fontweight="bold", color="white",
                zorder=3)

        # Divider line under title
        ax.plot([x + 0.2, x + BOX_W - 0.2],
                [y + BOX_H - 0.45, y + BOX_H - 0.45],
                color="white", linewidth=0.8, alpha=0.6, zorder=3)

        # Item boxes
        n_items   = len(items)
        available = BOX_H - 0.55 - pad         # height below divider
        spacing   = min(item_h, available / n_items - 0.04)
        start_y   = y + BOX_H - 0.55 - spacing

        for j, (label, item_col) in enumerate(items):
            iy = start_y - j * (spacing + 0.04)
            item_rect = mpatches.FancyBboxPatch(
                (x + pad, iy - spacing + 0.04),
                BOX_W - 2*pad, spacing,
                boxstyle="round,pad=0.04",
                facecolor=item_col, edgecolor="#aaaaaa",
                linewidth=0.6, zorder=2,
            )
            ax.add_patch(item_rect)
            ax.text(x + BOX_W/2, iy - spacing/2 + 0.04,
                    label,
                    ha="center", va="center",
                    fontsize=7.8, color="#1a1a1a", zorder=3)

    # ── Arrows ────────────────────────────────────────────────────────────
    arrow_kw = dict(arrowstyle="-|>", color=arrow_colour,
                    lw=1.8, mutation_scale=18)

    # S1 → S2 (horizontal top, left to right)
    x1e = positions["S1"][0] + BOX_W
    x2s = positions["S2"][0]
    mid_y_top = positions["S1"][1] + BOX_H / 2
    ax.annotate("", xy=(x2s, mid_y_top), xytext=(x1e, mid_y_top),
                arrowprops=arrow_kw)

    # S2 → S3 (vertical right, top to bottom)
    x_right = positions["S2"][0] + BOX_W / 2
    y2b     = positions["S2"][1]
    y3t     = positions["S3"][1] + BOX_H
    ax.annotate("", xy=(x_right, y3t), xytext=(x_right, y2b),
                arrowprops=arrow_kw)

    # S3 → S4 (horizontal bottom, right to left)
    x3s     = positions["S3"][0]
    x4e     = positions["S4"][0] + BOX_W
    mid_y_bot = positions["S3"][1] + BOX_H / 2
    ax.annotate("", xy=(x4e, mid_y_bot), xytext=(x3s, mid_y_bot),
                arrowprops=arrow_kw)

    # Step number badges on arrows
    arrow_labels = [
        (x1e + (x2s - x1e)/2, mid_y_top + 0.18, "①"),
        (x_right + 0.22,      (y2b + y3t)/2,    "②"),
        (x3s + (x4e - x3s)/2, mid_y_bot + 0.18, "③"),
    ]
    for lx, ly, lab in arrow_labels:
        ax.text(lx, ly, lab, ha="center", va="bottom",
                fontsize=9, color=arrow_colour, fontweight="bold")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=dataset_colour, edgecolor="#888",
                       label="Input datasets"),
        mpatches.Patch(facecolor=output_colour,  edgecolor="#888",
                       label="Output files"),
        mpatches.Patch(facecolor=process_colour, edgecolor="#888",
                       label="Processing steps"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              fontsize=8.5, framealpha=0.95,
              bbox_to_anchor=(0.99, 0.01))

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        "GUO Data Pipeline — Processing Workflow",
        fontsize=12, fontweight="bold", pad=14, y=0.98,
    )

    _save_fig(fig, "fig5_pipeline_workflow")


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURE S1 — PCI Distribution
# ═════════════════════════════════════════════════════════════════════════════

def plot_pci_distribution(pci_df: pd.DataFrame):
    """
    Supplementary Figure S1: PCI distribution across countries by category.
    Violin plots showing within-country sub-district variation.
    """
    if pci_df.empty:
        print("  ⚠ Suppl. Figure S1 skipped: PCI results not available.")
        return

    categories = list(pci_df["category"].unique()) if "category" in pci_df.columns else []
    if not categories:
        return

    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5),
                             sharey=False)
    if len(categories) == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        sub = pci_df[pci_df["category"] == cat].copy()
        if "pci_mean" not in sub.columns:
            continue

        sub = sub.sort_values("pci_mean", ascending=True)
        countries_sorted = sub["country"].tolist()
        values           = sub["pci_mean"].tolist()
        errors_hi        = (sub["pci_mean"] + sub.get("pci_std", 0)).tolist()
        errors_lo        = (sub["pci_mean"] - sub.get("pci_std", 0)).clip(lower=0).tolist()

        bars = ax.barh(range(len(countries_sorted)), values,
                       color="#74add1", alpha=0.85, edgecolor="#2c7bb6")

        # Error bars (std dev)
        ax.errorbar(
            values, range(len(countries_sorted)),
            xerr=[
                [v - lo for v, lo in zip(values, errors_lo)],
                [hi - v  for v, hi in zip(values, errors_hi)],
            ],
            fmt="none", color="#333333", capsize=3, linewidth=1,
        )

        ax.set_yticks(range(len(countries_sorted)))
        ax.set_yticklabels(countries_sorted, fontsize=8)
        ax.set_xlabel(f"Mean PCI\n(places per 1,000 residents)", fontsize=8)
        ax.set_title(f"{cat.capitalize()} facilities", fontweight="bold", fontsize=9)

        # Annotate bars
        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)

    fig.suptitle(
        "Supplementary Figure S1: Place Coverage Index by Service Category\n"
        "Mean ± SD across sub-districts in primary urban agglomeration",
        fontsize=9,
    )
    _save_fig(fig, "figS1_pci_distribution")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_figures():
    """Generate all five main figures and supplementary figure S1."""
    print("\n" + "="*65)
    print("  FIGURE GENERATION")
    print("="*65)

    # Load results files
    def safe_read(path, **kwargs):
        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            print(f"  ⚠ Not found: {Path(path).name} — some figures may be skipped.")
            return pd.DataFrame()

    bcr_df   = safe_read(TABLE_DIR / "bcr_bcc_results.csv")
    pga_df   = safe_read(TABLE_DIR / "pga_results.csv")
    pci_df   = safe_read(TABLE_DIR / "pci_results.csv")
    suit_df_path = TABLE_DIR / "sdg11_suitability_matrix.csv"

    suit_df = None
    try:
        suit_df = pd.read_csv(suit_df_path, index_col=0)
    except FileNotFoundError:
        pass

    gpkg_path = str(TABLE_DIR.parent / "gpkg" / "benchmark_metrics_subdistrict.gpkg")

    print("\n── Figure 1: Study Area Map ──")
    plot_study_area()

    print("\n── Figure 2: BCR + BCC Results ──")
    # Load MS Buildings BCR if available
    ms_bcr_path = TABLE_DIR.parent / "tables" / "ms_buildings_bcr.csv"
    if not ms_bcr_path.exists():
        ms_bcr_path = TABLE_DIR / "ms_buildings_bcr.csv"
    ms_bcr_df = pd.DataFrame()
    if ms_bcr_path.exists():
        try:
            ms_bcr_df = pd.read_csv(ms_bcr_path)
            print(f"  MS Buildings BCR loaded: {len(ms_bcr_df)} countries")
        except Exception:
            pass
    plot_bcr_results(bcr_df, ms_bcr_df=ms_bcr_df if not ms_bcr_df.empty else None)

    print("\n── Figure 3: Population Grid Alignment ──")
    plot_pga_scatterplots(pga_df, subdistrict_gpkg=gpkg_path)

    print("\n── Figure 4: SDG 11 Suitability Heatmap ──")
    plot_suitability_heatmap(suit_df)

    print("\n── Figure 5: Pipeline Workflow ──")
    plot_pipeline_workflow()

    print("\n── Supplementary Figure S1: PCI Distribution ──")
    plot_pci_distribution(pci_df)

    print(f"\n✓ All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    generate_all_figures()