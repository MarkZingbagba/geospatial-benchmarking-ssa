# Geospatial Benchmarking SSA

Reproducible Python pipeline for benchmarking open geospatial data stacks
for SDG 11 sub-national urban monitoring across Sub-Saharan Africa.

## Associated publication

Zingbagba, M. (2026). Benchmarking Open Geospatial Data Stacks for SDG 11
Sub-National Urban Monitoring in Sub-Saharan Africa. *Scientific Data*
(submitted).

Benchmark dataset deposited at: https://doi.org/10.5281/zenodo.19643277

## Study countries

Ghana (GHA), Côte d'Ivoire (CIV), Senegal (SEN), Nigeria (NGA),
Cameroon (CMR), Kenya (KEN), Tanzania (TZA)

## Datasets benchmarked

- Overture Maps Foundation (release 2025-01-22.0)
- Global Human Settlement Layer — GHSL R2023A
- WorldPop 2020 UN-adjusted constrained estimates
- Microsoft Global ML Building Footprints (2023)

## Metrics computed

| Metric | Description |
|--------|-------------|
| BCR | Building Completeness Ratio — % of GHSL built-up cells with ≥1 building centroid |
| BCC | Building Category Completeness — % of buildings with non-null class/subtype |
| PCI | Place Coverage Index — service facility POIs per 1,000 residents |
| PGA | Population Grid Alignment — GHS-POP vs WorldPop Pearson r and MAPE |

## Repository structure
## Installation

```bash
pip install -r requirements_analysis.txt
```

## Usage

```bash
# Check data availability
python run_analysis.py --step inventory

# Run full pipeline
python run_analysis.py

# Run for specific countries
python run_analysis.py --country GHA NGA KEN

# Run specific steps
python run_analysis.py --step bcr
python run_analysis.py --step pci
python run_analysis.py --step synthesis
python run_analysis.py --step figures
```

## Configuration

Before running, open `analysis_config.py` and update `BASE_DATA_DIR`
to point to your local data directory containing the downloaded
Overture Maps, GHSL, and WorldPop files.

## Requirements

- Python 3.12
- DuckDB 0.10+ (required for large GeoParquet files)
- See `requirements_analysis.txt` for full dependency list

## Licence

MIT — see [LICENSE](LICENSE)

## Contact

Mark Zingbagba  
Ghana Urban Observatory, Expertise France – AFD Group  
mark.zingbagba@expertisefrance.fr
