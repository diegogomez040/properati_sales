# Bogotá Property Price Predictor

## Overview

This project builds a machine learning pipeline that estimates residential property prices in Bogotá, Colombia from six basic listing characteristics: location, property type, area, bedrooms, bathrooms, and parking. The data was collected by scraping 5,002 listings from [Properati](https://www.properati.com.co), cleaned into ~4,509 structured records, and filtered to 3,133 residential listings used for modeling. The final model — a stacking ensemble of tuned Random Forest and XGBoost — achieves **RMSE ~391 M COP**, meeting the project target of < 400 M COP. The project is structured as three narrative-driven notebooks covering the full data science workflow from raw HTML scraping to an interactive price prediction tool.

---

## Project Structure

```
properati_sales/
├── notebooks/
│   ├── 01_data_cleaning.ipynb   # Web scraping reference + 10-step cleaning pipeline
│   ├── 02_eda.ipynb             # Market analysis: distributions, neighborhood tiers, correlations
│   └── 03_modeling.ipynb        # Feature engineering, model training, tuning, and prediction
├── data/                        # Gitignored — generated locally by running the notebooks
│   ├── properati.csv            # Raw scraped listings (5,002 rows)
│   └── cleaned_properati.csv    # Cleaned dataset produced by Notebook 1
└── README.md
```

> **Note:** The `data/` folder is excluded from version control. Running Notebook 1 end-to-end generates `cleaned_properati.csv`, which Notebooks 2 and 3 consume.

---

## Requirements

**Python:** 3.10+

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation and cleaning |
| `numpy` | Numerical operations and log transforms |
| `scikit-learn` | Preprocessing, model training, cross-validation, stacking |
| `xgboost` | Gradient boosting regressor |
| `matplotlib` | Charts and visualizations |
| `seaborn` | Statistical plots |
| `requests` + `beautifulsoup4` | Web scraping (reference only — data already collected) |

Install all dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests beautifulsoup4
```

---

## Usage

Run the notebooks in order. Each notebook produces an output consumed by the next.

**Step 1 — Data Cleaning**
```
notebooks/01_data_cleaning.ipynb
```
Parses raw scraped text into typed, model-ready columns. Exports `cleaned_properati.csv` to the `data/` folder.

**Step 2 — Exploratory Data Analysis**
```
notebooks/02_eda.ipynb
```
Visualizes the Bogotá real estate market. Requires `cleaned_properati.csv`. No file output — findings directly inform modeling decisions in Notebook 3.

**Step 3 — Modeling & Prediction**
```
notebooks/03_modeling.ipynb
```
Trains and evaluates all models. To predict a custom property price, edit the input cell in **Section 11** and re-run:

```python
# Edit these six values
bedrooms  = 3
bathrooms = 2
area_m2   = 120
parking   = 1              # 1 = Yes, 0 = No
location  = "Chapinero"
prop_type = "Apartamento"  # Apartamento | Casa | Apartaestudio
```

The model automatically engineers all derived features and outputs the estimated price in COP.

---

## Inputs & Outputs

| Notebook | Input | Output |
|---|---|---|
| `01_data_cleaning.ipynb` | `data/properati.csv` (5,002 raw rows) | `data/cleaned_properati.csv` (~4,509 clean rows) |
| `02_eda.ipynb` | `data/cleaned_properati.csv` | Charts and market insights (no file) |
| `03_modeling.ipynb` | `data/cleaned_properati.csv` | Trained models + price prediction |

---

## Methodology

### Data Collection
Listings were scraped from Properati using `requests` and `BeautifulSoup` across all property types: apartments, houses, offices, commercial spaces, and studio apartments. The scraper extracted 7 fields per listing: price, location, type, bedrooms, bathrooms, area, and parking.

### Cleaning (10-step pipeline)
All 7 fields arrived as raw HTML text requiring full parsing. Key steps include stripping currency symbols and converting prices to float, extracting numeric area values from strings like `"120 m²"`, imputing 2,318 null parking values as absence (0), and applying **95th-percentile outlier caps** on price and area to reduce the influence of extreme luxury listings that the 6-feature model cannot reliably predict.

### Feature Engineering
Four data-level changes applied before model training, each motivated by a specific EDA finding:

| Change | Motivated by |
|---|---|
| **Residential filter** — remove `Local`, `Oficina` | EDA §2: commercial listings price by floor space and location for revenue, not bedrooms and bathrooms |
| **Rare location grouping** — neighborhoods < 15 listings → `"Other"` | EDA §3: sparse neighborhoods produce unreliable target-encoding estimates |
| **Zone feature** — maps each neighborhood to 1 of 6 Bogotá macro-zones | EDA §3: the north-south price divide is consistent and needs a fallback signal for sparse neighborhoods |
| `bath_per_bed`, `log_area_per_bed`, `total_rooms`, `loc_count` | EDA §5–6: bathroom ratio and area-per-room capture quality and density signals that raw counts miss |

### Preprocessing Pipeline
Built with scikit-learn `ColumnTransformer`:
- **Numeric features** → `StandardScaler` → `PolynomialFeatures(degree=2, interaction_only=True)`
- **Location & zone** → `TargetEncoder` (mean log-price per category — direct price signal)
- **Property type** → `OneHotEncoder`

### Results

| Model | R² | RMSE (M COP) |
|---|---|---|
| Ridge | 0.633 | ~448 |
| XGBoost (baseline) | 0.685 | ~416 |
| Random Forest (baseline) | 0.702 | ~395 |
| Random Forest (tuned) | ~0.699 | ~397 |
| XGBoost (tuned) | ~0.704 | ~386 |
| **Stacking Ensemble** | **0.704** | **~391 ✅** |

**Targets:** RMSE < 400 M COP ✅ &nbsp;·&nbsp; R² ≥ 0.72 ❎

Hyperparameter tuning uses `RandomizedSearchCV` with `scoring="r2"` and 5-fold cross-validation (40 iterations per model). The stacking ensemble combines both tuned models as base learners with a `Ridge` meta-learner.

---

## Key Findings

- **Location drives up to 10× price differences.** North-zone neighborhoods (El Retiro, El Chicó, Los Rosales) have medians above 1.4 billion COP; south-zone neighborhoods (Usme, San Cristóbal, Ricaurte) fall below 400 million COP.
- **Area is the strongest individual numeric predictor** (Pearson r = 0.66 in log-space). A 1% increase in area associates with a ~0.68% price increase — sub-linear returns, meaning each additional m² adds less value as properties grow larger.
- **Bathrooms outperform bedrooms as a price signal** (r = 0.45 vs. r = 0.24). Bathroom count is a stronger proxy for overall unit quality and finish level.
- **Commercial listings follow a fundamentally different pricing logic.** Excluding `Local` and `Oficina` from training reduced RMSE substantially — they price by floor space and commercial demand, not by residential characteristics.

---

## Limitations

- **No time dimension** — the model does not account for market trends or price appreciation over time.
- **Limited feature set** — floor level, building age, amenities, and proximity to public transit (TransMilenio) are absent from the scraped data.
- **Geographic sparsity** — approximately 40% of neighborhoods have fewer than 15 listings; predictions for these areas carry higher uncertainty and fall back to the macro-zone signal.
- **Bogotá only** — the pipeline is trained exclusively on Bogotá listings and is not intended for other Colombian cities without retraining.
- **R² target not met** — the final stacking ensemble reaches R² = 0.704, below the 0.72 target. The remaining gap is likely attributable to the limited feature set rather than model architecture.

---

## Author

Built as a portfolio project demonstrating end-to-end data science: web scraping, data cleaning, exploratory analysis, and machine learning with scikit-learn and XGBoost.
