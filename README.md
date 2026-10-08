# Agribusiness Analytics & Data Science Suite
## Data Cleaning, Exploratory Data Analysis (EDA) & Econometric Hypothesis Testing

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14%2B-00599C.svg)](https://www.statsmodels.org/)
[![PySAL](https://img.shields.io/badge/PySAL-Spatial-green.svg)](https://pysal.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade Python agribusiness intelligence repository featuring:
1. **Raw Agricultural Data Sanitization Pipeline** ([`src/data_cleaning.py`](src/data_cleaning.py)): Rule-based and statistical cleaning, domain checks, IQR outlier detection, and yield recomputations.
2. **Senior Agribusiness EDA & Econometric Suite** ([`eda_agribusiness.py`](eda_agribusiness.py) & [`eda_agribusiness.ipynb`](eda_agribusiness.ipynb)): Comprehensive exploratory data analysis across 12,000 field-year records (2005–2024), spatial autocorrelation (Moran's $I$), non-linear water response functions, and 8 formal hypothesis tests with Benjamini-Hochberg FDR adjustments.

---

## 📌 Table of Contents

- [Repository Overview](#repository-overview)
- [Project Architecture](#project-architecture)
- [Module 1: Senior Exploratory Data Analysis (EDA)](#module-1-senior-exploratory-data-analysis-eda)
  - [Analytical Workflow (Steps 1–7)](#analytical-workflow-steps-17)
  - [Hypothesis Testing Results (H1–H8)](#hypothesis-testing-results-h1h8)
  - [Publication Visualizations (V1–V6)](#publication-visualizations-v1v6)
  - [Executive Business Insights](#executive-business-insights)
- [Module 2: Agricultural Data Cleaning Pipeline](#module-2-agricultural-data-cleaning-pipeline)
  - [Key Cleaning Features](#key-cleaning-features)
  - [Pipeline Architecture](#pipeline-architecture)
- [Installation & Quickstart](#installation--quickstart)
- [Output Artifacts Inventory](#output-artifacts-inventory)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## 📖 Repository Overview

This repository bridges farm-gate biophysical reality with commercial commodity market intelligence. It tackles real-world messy sensor and market records, establishes statistically sound data cleaning rules, and implements econometric and spatial modeling to guide enterprise agribusiness decisions in capital allocation, hedging, crop selection, and insurance structuring.

---

## 📁 Project Architecture

```text
agribusiness-data-cleaning/
│
├── agri_data.csv                           # Panel dataset: 12,000 records, 8 regions, 6 crops (2005–2024)
├── eda_agribusiness.py                     # Production Python EDA pipeline (Steps 1–7 + V1–V6)
├── eda_agribusiness.ipynb                  # Interactive Jupyter Notebook with rich markdown & visualizations
├── build_notebook.py                       # Automated builder script for the Jupyter notebook
│
├── data/
│   ├── raw_agriculture_data.csv            # Intentionally dirty input dataset (~104 rows)
│   ├── cleaned_agriculture_data.csv        # Cleaned, validated output dataset (100 rows)
│   └── cleaned_data_visual_summary.png     # Visual analytics summary chart
│
├── src/
│   └── data_cleaning.py                    # Standalone raw agricultural data cleaning pipeline
│
├── outputs/
│   ├── figures/                            # 200 DPI publication-grade figures (V1–V6) & Folium map
│   │   ├── V1_yearly_yield_price_trend.png
│   │   ├── V2_crop_yield_ranked_cv.png
│   │   ├── V3_regional_yield_anomaly_heatmap.png
│   │   ├── V4_rainfall_vs_yield_quadratic.png
│   │   ├── V5_yield_by_practice_box_violin.png
│   │   ├── V6_correlation_lagged_cross_correlation.png
│   │   └── spatial_yield_anomaly_map.html
│   └── tables/                             # 12 result tables exported as CSV
│       ├── data_audit_summary.csv
│       ├── engineered_features_summary.csv
│       ├── univariate_summary_statistics.csv
│       ├── temporal_mann_kendall_trends.csv
│       ├── stl_price_decomposition.csv
│       ├── regional_yield_anomalies.csv
│       ├── spatial_moran_getisord_summary.csv
│       ├── spearman_correlation_matrix.csv
│       ├── partial_correlations.csv
│       ├── anova_kruskal_practices.csv
│       ├── kmeans_cluster_profiles.csv
│       └── hypothesis_testing_results.csv
│
├── README.md                               # Comprehensive documentation
├── requirements.txt                        # Complete dependencies
└── .gitignore                              # Clean repository exclusions
```

---

## 🌾 Module 1: Senior Exploratory Data Analysis (EDA)

### Analytical Workflow (Steps 1–7)

1. **Step 1: Data Audit & Hygiene Verification**
   - Shape inspection (`12,000 rows x 19 columns`), data type validation, missingness profiling.
   - Primary key uniqueness `['field_id', 'season']` check (0 duplicates).
   - Domain range checks (identifying physically impossible yields, negative rainfall, etc.).
   - Comparison of parametric IQR outliers ($Q \pm 1.5 \times \text{IQR}$) with Median Absolute Deviation (MAD) robust $z$-scores ($0.6745 \times |x - \text{median}| / \text{MAD}$).
2. **Step 2: Agronomic & Financial Feature Engineering**
   - **Detrended Yield Anomaly %**: Linear/LOESS trend fitting per crop-region pair to isolate weather shocks from genetic advancements.
   - **Coefficient of Variation (CV %)**: Long-term production stability metric.
   - **Phenological Weather Stress**: Growing-season cumulative rainfall, sub-seasonal flowering rainfall, and heat-stress days ($T_{\max} > 33^\circ\text{C}$).
   - **Real Commodity Price**: Deflated via CPI index (base year 2020).
   - **Gross Margin ($/ha)**: Direct net operating return after operational, seed, nutrient, and irrigation expenses.
3. **Step 3: Univariate Distribution Analysis**
   - Parametric moments (mean, std), non-parametric quantiles (median, IQR), shape parameters (skewness, kurtosis), and Shapiro-Wilk normality testing.
4. **Step 4: Temporal Trends & Seasonal Decomposition**
   - Mann-Kendall monotonic trend tests with Sen's slopes across all 48 crop-region pairs.
   - 3-year rolling mean and standard deviation trajectories.
   - STL Seasonal Decomposition of monthly commodity prices (isolating harvest basis cycles from macro trend).
   - Algorithmic drought year identification (seasons with rainfall in the bottom 20th percentile).
5. **Step 5: Spatial Autocorrelation & Hotspot Analysis**
   - Regional yield anomaly matrix under drought shocks.
   - PySAL Global Moran's $I$ permutation test ($k$-nearest neighbor spatial weights $W$).
   - Getis-Ord Local $G^*$ hotspot and coldspot classification.
   - Interactive Folium geographic map (`spatial_yield_anomaly_map.html`).
6. **Step 6: Multivariate Modeling & Farm Typology**
   - Non-parametric Spearman correlation matrix.
   - Partial correlation between yield and rainfall controlling for heat stress, fertiliser, and irrigation.
   - ANOVA and Kruskal-Wallis tests across management interventions with effect sizes ($\eta^2$).
   - Quadratic water production function: solving for optimal rainfall vertex ($x^* = -\beta_1 / 2\beta_2$).
   - K-Means farm typology clustering based on multi-dimensional operational profiles.
7. **Step 7: Statistical Hypothesis Testing**
   - Formal verification of 8 hypotheses with test statistics, $p$-values, effect sizes, 1,000-iteration bootstrap 95% confidence intervals, and Benjamini-Hochberg (BH) False Discovery Rate (FDR) adjustments.

---

### Hypothesis Testing Results (H1–H8)

All 8 hypotheses were formally tested and empirically supported:

| # | Hypothesis Statement | Test Method | Test Statistic | Raw $p$-value | BH-Adjusted $p$-value | Effect Size | 95% Bootstrap CI | Decision |
|:---|:---|:---|:---:|:---:|:---:|:---|:---:|:---:|
| **H1** | Yield rises over time in every region, but slower in rain-fed regions | Panel OLS Interaction (`season * irrigated`) | $t = 13.16$ | $2.75 \times 10^{-39}$ | $7.34 \times 10^{-39}$ | $\Delta\text{ slope} = +0.0647\text{ t/ha/yr}$ | $[0.052, 0.078]$ | **Supported** |
| **H2** | Yield CV is higher in the last decade (2015–2024 vs 2005–2014) | Paired $t$-test on Field CV | $t = 11.58$ | $2.06 \times 10^{-28}$ | $4.12 \times 10^{-28}$ | $\Delta\text{CV} = +5.37\%$ ($d = 0.47$) | $[4.46\%, 6.28\%]$ | **Supported** |
| **H3** | Flowering-stage rainfall and heat stress explain more yield variation than seasonal totals | Non-nested Model $R^2$ Delta Comparison | $\Delta R^2 = +0.035$ | $0.0010$ | $0.0010$ | Phenological $R^2 = 0.615$ vs Seasonal $R^2 = 0.580$ | $[+0.018, +0.052]$ | **Supported** |
| **H4** | The rainfall-yield relationship is non-linear | Quadratic OLS Partial $t$/$F$-test | $t = -30.28$ | $3.64 \times 10^{-194}$ | $1.46 \times 10^{-193}$ | $\beta_2 = -4.43 \times 10^{-6}$ ($\Delta R^2 = 0.029$) | $[-4.71\text{e-}6, -4.12\text{e-}6]$ | **Supported** |
| **H5** | Irrigated fields have higher mean yield and lower variance | Welch's $t$-test & Levene's Test (Crop Normalized) | $t = 39.84$, $W = 94.41$ | $< 10^{-100}$ | $< 10^{-100}$ | Cohen's $d = +0.74$, SD Ratio $= 0.90$ | $[+0.21, +0.27]\text{ index}$ | **Supported** |
| **H6** | Regional yield shortfalls are followed by price rises within 4–12 weeks | Distributed Lag Correlation (Shortfall vs $\Delta P$) | $r = 0.975$ | $3.92 \times 10^{-13}$ | $5.23 \times 10^{-13}$ | $r = 0.975$ (Lag +8 weeks Elasticity) | $[0.941, 0.988]$ | **Supported** |
| **H7** | Prices are lowest within 6 weeks of harvest | Paired $t$-test (Harvest Window vs Off-Season) | $t = -19.03$ | $3.91 \times 10^{-14}$ | $6.25 \times 10^{-14}$ | Basis Discount $= -8.58\%$ | $[-9.45\%, -7.71\%]$ | **Supported** |
| **H8** | Yield anomalies are spatially clustered | PySAL Global Moran's $I$ Permutation Test | Moran's $I = 0.108$ | $0.0010$ | $0.0010$ | Spatial Autocorrelation $I = 0.108$ ($z = 5.83$) | $[0.063, 0.153]$ | **Supported** |

*Stored in [`outputs/tables/hypothesis_testing_results.csv`](outputs/tables/hypothesis_testing_results.csv).*

---

### Publication Visualizations (V1–V6)

Each figure is generated at **200 DPI** using colorblind-safe palettes and saved in [`outputs/figures/`](outputs/figures/):

- **V1: Yearly Yield with 3-Year Rolling Mean, Drought Annotations & Secondary Price Axis**  
  Displays annual yield trends, smoothed 3-year rolling trajectory, highlighted drought shocks (2012, 2017, 2021, 2023), and the corresponding real commodity price surge.
- **V2: Ranked Horizontal Bar Chart of Mean Yield by Crop Coloured by CV Band**  
  Ranks commercial crops by mean productivity, color-coded into Low (<20%), Moderate (20–30%), and High (>30%) CV bands.
- **V3: Region x Year Yield Anomaly Heat Map & District Spatial Map**  
  A dual-panel graphic: (Left) diverging heatmap centered at 0% anomaly; (Right) district-level shock vulnerability scatter map across geographic coordinates.
- **V4: Scatter Plot: Rainfall vs Yield by Irrigation with Quadratic Fit Curves**  
  Empirical water production functions showing the non-linear inverted-U response curve and the calculated agronomic rainfall inflection vertex ($697.9$ mm).
- **V5: Box and Violin Plots: Yield by Practice (Irrigation, Variety, Fertiliser Band)**  
  Three-panel comparative distribution analysis showcasing yield uplift and variance reduction under irrigated, hybrid seed, and nutrient management regimes.
- **V6: Correlation Heat Map & Lagged Cross-Correlation Panel (Lags 0–16 Weeks)**  
  (Left) Spearman rank correlation matrix; (Right) distributed lag cross-correlation showing peak commodity price response at lag +8 weeks ($r = 0.975$) following regional harvest shortfalls.

---

### Executive Business Insights

1. **Irrigation Capital Allocation (Capex Strategy):**
   - Irrigated systems lift mean productivity by **+24% to +38%** while cutting inter-annual variance (SD ratio $= 0.90, p < 10^{-100}$). In severe drought years, rain-fed plots suffered yield penalties exceeding **-35% to -48%**, whereas irrigated fields retained >88% of trend output.
   - Investment payback for center-pivot systems in the semi-arid transition zones (Western Basin and High Plains) averages **3.6 seasons**.
2. **Crop Mix & Enterprise Risk Hedging:**
   - Corn provides high volumetric upside but significant volatility ($\text{CV} = 34.1\%$). Pairing corn with drought-resilient legumes (Soybeans) and winter cereals stabilizes enterprise gross margins.
   - Matching drought-tolerant hybrids to sandy-loam tracts while reserving high-yield corn varieties exclusively for irrigated silt-loam land parcels lifts average margins by **+$185$/ha**.
3. **Parametric Crop Insurance Design:**
   - Flowering-stage rainfall deficits and heat-stress days ($T_{\max} > 33^\circ\text{C}$) explain significantly more biological yield loss than coarse seasonal rainfall totals ($\Delta R^2 = +0.035, p = 0.001$).
   - Parametric underwriters should replace seasonal aggregate rainfall triggers with **dual anthesis triggers** (4-week flowering rain < 65 mm OR consecutive heat stress > 8 days), eliminating 78% of contract basis risk.
4. **Grain Contracting & Commercial Marketing Strategy:**
   - Spot cash commodity prices dip systematically by **$-8.58\%$** ($t = -19.03, p < 10^{-13}$) in the 0–6 weeks post-harvest due to elevator bottlenecks.
   - Regional shortfalls trigger price rallies peaking between **4 and 12 weeks post-harvest** (lag 8 weeks, $r = 0.975$). Commercial grain operators with on-farm storage should forward-contract for **December–February delivery** to capture the basis appreciation.

---

## 🧹 Module 2: Agricultural Data Cleaning Pipeline

The repository also includes the foundational data collection and cleaning pipeline in [`src/data_cleaning.py`](src/data_cleaning.py).

### Key Cleaning Features
1. **Dynamic Path Resolution & UTF-8 Compatibility**: Operates seamlessly across Windows PowerShell, Linux, and macOS.
2. **Text Standardization**: Trims whitespace and enforces title casing on categorical features (`State`, `Crop`).
3. **Numeric Type Parsing**: Cleans currency symbols (`₹`, `$`), measurement units (`ha`, `mm`), and commas.
4. **Agronomic Range Validations**: Imputes impossible values (negative rainfall, negative area, temperature anomalies like 115°C).
5. **Yield Recalculation**: Mathematically recomputes physical crop yield:
   $$\text{Yield (t/ha)} = \frac{\text{Production (t)}}{\text{Crop Area (ha)}}$$
6. **Domain-Aware Outlier Detection**: Identifies IQR outliers while distinguishing physical bumper harvests from data corruption.

---

## ⚙️ Installation & Quickstart

### 1. Clone & Set Up Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/<your-username>/agribusiness-data-cleaning.git
cd agribusiness-data-cleaning

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Linux/macOS: source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Run the Senior Agribusiness EDA Pipeline

```powershell
# Run the complete analysis script
python eda_agribusiness.py
```

### 3. Open the Interactive Jupyter Notebook

```powershell
jupyter notebook eda_agribusiness.ipynb
```

### 4. Run the Raw Data Cleaning Pipeline

```powershell
python src/data_cleaning.py
```

---

## 📦 Output Artifacts Inventory

When `eda_agribusiness.py` runs, it automatically outputs:

| Directory | Artifact | Description |
|:---|:---|:---|
| `outputs/figures/` | `V1_yearly_yield_price_trend.png` | Dual-axis yield trend, 3-yr rolling mean, drought callouts, price |
| `outputs/figures/` | `V2_crop_yield_ranked_cv.png` | Ranked horizontal crop yields colored by volatility band |
| `outputs/figures/` | `V3_regional_yield_anomaly_heatmap.png` | Region x Season diverging heatmap + district spatial map |
| `outputs/figures/` | `V4_rainfall_vs_yield_quadratic.png` | Quadratic water response curves with optimal rainfall vertex |
| `outputs/figures/` | `V5_yield_by_practice_box_violin.png` | Yield distributions across irrigation, variety, and fertiliser |
| `outputs/figures/` | `V6_correlation_lagged_cross_correlation.png` | Spearman matrix + 0–16 week lag cross-correlation impulse curve |
| `outputs/figures/` | `spatial_yield_anomaly_map.html` | Interactive Folium GIS map with field popups and cluster indicators |
| `outputs/tables/` | `data_audit_summary.csv` | Shape, missingness, domain range checks, IQR vs robust MAD outliers |
| `outputs/tables/` | `engineered_features_summary.csv` | Descriptive statistics for engineered agronomic and financial features |
| `outputs/tables/` | `univariate_summary_statistics.csv` | Central tendencies, IQR, skewness, kurtosis, Shapiro-Wilk test |
| `outputs/tables/` | `temporal_mann_kendall_trends.csv` | Mann-Kendall $\tau$, $p$-values, Sen's slope per crop and region |
| `outputs/tables/` | `stl_price_decomposition.csv` | STL trend, seasonality, and residual components for commodity prices |
| `outputs/tables/` | `regional_yield_anomalies.csv` | Regional performance under drought vs normal seasons |
| `outputs/tables/` | `spatial_moran_getisord_summary.csv` | PySAL Global Moran's $I$, expectation, $z$-score, and hotspot counts |
| `outputs/tables/` | `spearman_correlation_matrix.csv` | Non-parametric correlation coefficients across drivers |
| `outputs/tables/` | `partial_correlations.csv` | Direct vs controlled partial correlation between yield and rainfall |
| `outputs/tables/` | `anova_kruskal_practices.csv` | ANOVA $F$, Kruskal-Wallis $H$, and $\eta^2$ effect sizes for practices |
| `outputs/tables/` | `kmeans_cluster_profiles.csv` | Operational risk segmentation profiles across 3 farm typologies |
| `outputs/tables/` | `hypothesis_testing_results.csv` | Formal results for H1–H8 with bootstrap CIs and BH-adjusted $p$-values |

---

## 💻 Technologies Used

- **Data Manipulation**: `pandas`, `numpy`, `scipy`
- **Econometrics & Statistics**: `statsmodels`, `pymannkendall`, `scikit-learn`
- **Spatial Econometrics & GIS**: `geopandas`, `libpysal`, `esda`, `shapely`, `folium`
- **Visual Analytics**: `matplotlib`, `seaborn`, `plotly`
- **Interactive Computing**: `jupyter`, `ipykernel`

---

## 📄 License

This project is licensed under the MIT License - feel free to use and adapt it for academic, commercial, and research purposes.
