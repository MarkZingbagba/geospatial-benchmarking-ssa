"""
=============================================================================
run_analysis.py — Master Analysis Runner 
Benchmarking Open Geospatial Data Stacks for SDG 11
=============================================================================
Executes the full analysis pipeline in sequence:

  Step 1 — Inventory: check all data files are present
  Step 2 — BCR + BCC: building completeness (bcr_bcc_analysis.py)
  Step 3 — PCI + PGA: place coverage + population alignment (pci_pga_analysis.py)
  Step 4 — Synthesis: compile tables + suitability matrix (synthesis_suitability.py)
  Step 5 — Figures:   generate all publication figures (figures.py)

Usage:
  python run_analysis.py                    # full pipeline
  python run_analysis.py --step inventory   # data check only
  python run_analysis.py --step bcr         # BCR + BCC only
  python run_analysis.py --step pci         # PCI + PGA only
  python run_analysis.py --step synthesis   # compile outputs only
  python run_analysis.py --step figures     # figures only
  python run_analysis.py --country GHA NGA  # specific countries
=============================================================================
"""

import argparse
import sys
import logging
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from analysis_config import LOG_DIR, COUNTRIES, TABLE_DIR

logging.basicConfig(
    filename=str(LOG_DIR / "run_analysis.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def run_inventory(country_filter=None):
    from data_loader import print_data_inventory, check_data_availability

    print("\n" + "="*65)
    print("  STEP 0 — DATA INVENTORY")
    print("="*65)

    print_data_inventory()

    # Check critical files exist before proceeding
    all_ok   = True
    iso_list = country_filter or list(COUNTRIES.keys())

    for iso3 in iso_list:
        status = check_data_availability(iso3)
        if not status.get("exists"):
            print(f"  ✗ {iso3}: folder not found — {status['folder']}")
            all_ok = False
        if "✗" in status.get("buildings_parquet", ""):
            print(f"  ✗ {iso3}: buildings.parquet missing")
            all_ok = False
        if "✗" in status.get("ghsl_GHS_BUILT_S", ""):
            print(f"  ⚠ {iso3}: GHS_BUILT_S raster missing — BCR will fail")
        if "✗" in status.get("worldpop", ""):
            print(f"  ⚠ {iso3}: WorldPop raster missing — PGA/PCI denominators unavailable")

    if all_ok:
        print("\n  ✓ All required data files found. Ready to proceed.")
    else:
        print("\n  ✗ Some files are missing. Check paths in analysis_config.py")
        print("    and re-run the download pipeline if needed.")
    return all_ok


def run_bcr_step(country_filter=None):
    from bcr_bcc_analysis import run_bcr_bcc_all_countries

    if country_filter:
        # Temporarily filter COUNTRIES in config
        import analysis_config as cfg
        original = dict(cfg.COUNTRIES)
        cfg.COUNTRIES = {k: v for k, v in original.items() if k in country_filter}

    result = run_bcr_bcc_all_countries()

    if country_filter:
        cfg.COUNTRIES = original

    return result


def run_pci_step(country_filter=None):
    from pci_pga_analysis import run_pci_pga_all_countries

    if country_filter:
        import analysis_config as cfg
        original = dict(cfg.COUNTRIES)
        cfg.COUNTRIES = {k: v for k, v in original.items() if k in country_filter}

    result = run_pci_pga_all_countries()

    if country_filter:
        cfg.COUNTRIES = original

    return result


def run_synthesis_step():
    from synthesis_suitability import run_synthesis
    return run_synthesis()


def run_figures_step():
    from figures import generate_all_figures
    generate_all_figures()


def check_requirements():
    """
    Verify all required Python packages are installed before running.
    """
    required = {
        "duckdb":         "pip install duckdb>=0.10",
        "geopandas":      "pip install geopandas>=0.14",
        "rasterio":       "pip install rasterio>=1.3",
        "numpy":          "pip install numpy",
        "pandas":         "pip install pandas",
        "scipy":          "pip install scipy",
        "matplotlib":     "pip install matplotlib",
        "shapely":        "pip install shapely>=2.0",
        "pyproj":         "pip install pyproj",
    }
    optional = {
        "exactextract":   "pip install exactextract   # preferred for zonal stats",
        "rasterstats":    "pip install rasterstats     # fallback for zonal stats",
        "mercantile":     "pip install mercantile      # needed for MS Buildings tiles",
    }

    print("\n── Package check ──")
    all_ok = True
    for pkg, install_cmd in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — REQUIRED: {install_cmd}")
            all_ok = False

    for pkg, install_cmd in optional.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} (optional)")
        except ImportError:
            print(f"  ⚠ {pkg} — optional: {install_cmd}")

    if not all_ok:
        print("\n  Install missing packages before proceeding.")
        print("  Run: pip install -r requirements_analysis.txt\n")
    return all_ok


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Analysis Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py                         # full pipeline
  python run_analysis.py --step inventory        # check data files only
  python run_analysis.py --step bcr              # BCR + BCC only
  python run_analysis.py --step pci              # PCI + PGA only
  python run_analysis.py --step synthesis        # compile tables only
  python run_analysis.py --step figures          # generate figures only
  python run_analysis.py --country GHA NGA KEN   # specific countries only
  python run_analysis.py --check                 # check Python packages
        """,
    )
    parser.add_argument(
        "--step",
        choices=["all", "inventory", "bcr", "pci", "synthesis", "figures"],
        default="all",
        help="Which analysis step to run (default: all)",
    )
    parser.add_argument(
        "--country",
        nargs="+",
        choices=list(COUNTRIES.keys()),
        default=None,
        help="Limit analysis to specific ISO3 codes (e.g. GHA NGA)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Python package requirements and exit",
    )
    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="Skip inventory check (use when re-running after confirmed data)",
    )

    args = parser.parse_args()

    # Package check
    if args.check:
        check_requirements()
        sys.exit(0)

    print("\n" + "█"*65)
    print("  PAPER — BENCHMARK ANALYSIS PIPELINE")
    print("  Benchmarking Open Geospatial Data Stacks for SDG 11")
    print("  Sub-National Urban Monitoring in Sub-Saharan Africa")
    print("█"*65)

    # Always check packages first
    check_requirements()

    countries = args.country  # None = all countries

    if args.step in ["all", "inventory"] and not args.skip_inventory:
        ok = run_inventory(countries)
        if not ok and args.step == "all":
            print("\n  Stopping: fix data issues before running full pipeline.")
            sys.exit(1)

    if args.step in ["all", "bcr"]:
        print("\n" + "─"*65)
        print("  STEP 1 — BCR + BCC ANALYSIS")
        print("─"*65)
        bcr_df = run_bcr_step(countries)

    if args.step in ["all", "pci"]:
        print("\n" + "─"*65)
        print("  STEP 2 — PCI + PGA ANALYSIS")
        print("─"*65)
        run_pci_step(countries)

    if args.step in ["all", "synthesis"]:
        print("\n" + "─"*65)
        print("  STEP 3 — SYNTHESIS + SUITABILITY MATRIX")
        print("─"*65)
        run_synthesis_step()

    if args.step in ["all", "figures"]:
        print("\n" + "─"*65)
        print("  STEP 4 — FIGURE GENERATION")
        print("─"*65)
        run_figures_step()

    print("\n" + "█"*65)
    print("  PIPELINE COMPLETE")
    print(f"  Tables  → {TABLE_DIR}")
    print(f"  Figures → {TABLE_DIR.parent / 'figures'}")
    print(f"  GPKG    → {TABLE_DIR.parent / 'gpkg'}")
    print("█"*65 + "\n")
