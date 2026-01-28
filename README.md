# Florida Hurricane Preparedness Algorithm

A Python-based system that ranks Florida counties by hurricane evacuation and preparedness risk using historical storm data, population metrics, and electrical grid vulnerability.

Designed as a decision-support tool for emergency planning that accounts for **actual hurricane paths**, not just static risk scores.

---

## Why This Project

Florida hurricanes affect counties unevenly based on storm trajectory, population density, and infrastructure stress. Many existing tools rely on static risk maps that do not consider the specific path or intensity of a hurricane.

This project evaluates county-level risk dynamically using quantitative modeling and publicly available data.

---

## Key Features

- Uses historical hurricane paths (Hurricanes Fay, Idalia, and Irma)
- Computes storm-to-county distance using the Haversine formula
- Normalizes heterogeneous data (population, wind speed, energy usage)
- Applies Principal Component Analysis (PCA) to derive objective weights
- Produces a ranked list of counties by evacuation priority
- Exports results to CSV and visualizes risk via heatmaps

---

## Data Sources

- **NOAA** – Historical hurricane tracks (coordinates, wind speed)
- **U.S. Census (2021)** – County population density
- **U.S. Energy Information Administration (EIA)** – Electricity generation and demand
- **FEMA / National Risk Index** – Supplemental risk indicators

Each of Florida’s 67 counties is represented by a central geographic coordinate.

---

## How It Works

1. Ingest data from multiple public datasets  
2. Compute hurricane proximity using great-circle distance  
3. Classify storm intensity using the Saffir–Simpson scale  
4. Normalize all variables to a 0–1 range  
5. Apply PCA to determine factor weights  
6. Calculate a weighted risk score for each county  
7. Rank counties by overall preparedness risk  

---

## Algorithms & Methods

- **Haversine Formula** – Accurate geographic distance calculation
- **Min–Max Normalization** – Unit comparability across datasets
- **Principal Component Analysis (PCA)** – Data-driven weight derivation
- **Weighted Scoring Model** – Final county ranking
- **Pandas Sorting (Timsort)** – Ordered results

---

## Project Structure

```text
├── data/        # Raw and processed datasets
├── src/         # Core analysis and algorithms
├── results/     # CSV outputs and visualizations
├── main.py      # Entry point
└── requirements.txt
```
## Running the Project

```bash
pip install -r requirements.txt
python main.py
```
## Authors

Thea Hird

Joseph Ruminjo

Samantha Cotugno
