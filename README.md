# Data Collection and Cleaning for Agribusiness Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c.svg)](https://matplotlib.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, beginner-friendly Python data pipeline designed for agribusiness and agricultural data science. This project simulates real-world agricultural field data, detects real-life data quality flaws, implements systematic cleaning and domain-specific imputation, detects potential outliers via the Interquartile Range (IQR) technique, validates integrity, and produces a clean dataset ready for downstream data visualization and analytics.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Objective](#objective)
- [Key Features](#key-features)
- [Data Pipeline Architecture](#data-pipeline-architecture)
- [Dataset Description](#dataset-description)
- [Data Quality Problems Intentionally Injected](#data-quality-problems-intentionally-injected)
- [Cleaning Methodology](#cleaning-methodology)
- [Outlier Treatment in Agribusiness](#outlier-treatment-in-agribusiness)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [How to Run](#how-to-run)
- [Example Output](#example-output)
- [Future Improvements](#future-improvements)

---

## 📖 Project Overview

Agricultural datasets collected from farm sensors, manual field surveys, mandi market records, and meteorological stations frequently suffer from formatting anomalies, sensor glitches, manual input errors, and missing metrics. 

This repository provides an end-to-end data cleaning pipeline that converts messy raw agricultural data into validated, analysis-ready tabular data without relying on complex black-box libraries.

---

## 🎯 Objective

- Demonstrate core Data Science Engineering data collection, inspection, cleaning, and validation workflows.
- Apply rule-based and statistical cleaning techniques (handling duplicates, trimming whitespace, standardizing categorical casing, parsing irregular strings).
- Implement domain-specific corrections (recomputing physical crop yield, filtering negative area/rainfall/production values).
- Apply statistical outlier detection (IQR method) while respecting domain realities (e.g., distinguishing bumper harvests from data corruption).
- Produce reproducible, high-integrity clean data for visualization and exploratory data analysis (EDA).

---

## ✨ Key Features

1. **Dual Console & Script Compatibility**: Dynamic path resolution and automatic UTF-8 encoding support for Windows PowerShell, CMD, Linux, and macOS.
2. **Text Standardization**: Automatic trimming of leading/trailing whitespaces and conversion to standardized Title Case across states and crops.
3. **Data Type Correction**: Robust string parsing to strip currency symbols (`₹`, `$`), measurement units (`ha`, `mm`), and commas from numeric fields.
4. **Domain-Specific Validation**:
   - Rejection and imputation of impossible values (negative rainfall, negative crop area, negative production, extreme temperatures like 115°C or -45°C).
   - Mathematical recalculation of `Yield_Tonnes_Per_Hectare` based on `Production_Tonnes / Crop_Area_Hectares`.
5. **Statistical Imputation**: Median imputation for numerical fields (resilient to skewness) and mode imputation for categorical attributes.
6. **Outlier Identification via IQR**: Identifies statistical outliers using the $1.5 \times \text{IQR}$ rule and provides agricultural domain reasoning rather than blindly deleting data.
7. **Automated Validation & Reporting**: Validates 0 remaining nulls, 0 duplicate rows, and valid physical constraints, accompanied by before-and-after summary statistics and a summary chart.

---

## 🔄 Data Pipeline Architecture

```text
  Raw Agricultural Data (.csv)
              │
              ▼
       [Data Collection]
      (Load via Pandas)
              │
              ▼
       [Data Inspection]
   (Inspect Shape, dtypes,
    Missing Values, Duplicates)
              │
              ▼
        [Data Cleaning]
   (Remove Duplicates, Clean Strings,
    Parse Numbers, Handle Negatives,
    Impute Missing, Recalculate Yield)
              │
              ▼
      [Data Validation]
   (Verify 0 Nulls, 0 Duplicates,
    Logical Bounds, IQR Outlier Check)
              │
              ▼
       [Clean Dataset]
  (Save clean CSV & Visuals)
              │
              ▼
 Ready for Agribusiness Analytics!
```

---

## 📊 Dataset Description

The dataset simulates multi-state Indian agricultural production records across major food and commercial crops (Wheat, Rice, Sugarcane, Cotton, Mustard, Maize, Soybean, and Pulses).

| Column Name | Description | Example Values |
| :--- | :--- | :--- |
| `Date` | Recording date | `2023-01-10`, `2023-04-18` |
| `State` | State of cultivation | `Punjab`, `Maharashtra`, `Gujarat` |
| `Crop` | Type of crop grown | `Wheat`, `Sugarcane`, `Cotton` |
| `Crop_Area_Hectares` | Total cultivated area in hectares | `150.0`, `280.0`, `320.5` |
| `Production_Tonnes` | Total agricultural output in metric tonnes | `675.0`, `21000.0`, `1280.0` |
| `Yield_Tonnes_Per_Hectare` | Crop productivity (Tonnes per Hectare) | `4.50`, `75.00`, `1.80` |
| `Rainfall_mm` | Cumulative rainfall during growing season (mm) | `65.5`, `180.0`, `220.0` |
| `Temperature_C` | Mean ambient seasonal temperature (°C) | `14.2`, `24.5`, `31.0` |
| `Market_Price_Per_Tonne` | Average wholesale mandi price (INR / Tonne) | `2250.0`, `3150.0`, `6400.0` |

---

## ⚠️ Data Quality Problems Intentionally Injected

To replicate real-world data collection issues, the raw file contains deliberate defects:

1. **Duplicate Records**: 4 exact duplicate rows injected across seasons.
2. **Text Formatting Irregularities**:
   - Inconsistent casing (e.g., `"punjab"`, `"MAHARASHTRA"`, `"wheat"`, `"COTTON"`).
   - Extra leading and trailing spaces (e.g., `" Haryana "`, `"  Rice"`).
3. **Incorrect Data Types**:
   - Currency symbols and commas embedded in numeric strings (`"₹2,300"`, `"$2300"`).
   - Units embedded inside numeric values (`"450 ha"`, `"22,500.0"`).
   - Text placeholder strings (`"N/A"`, `"missing"`).
4. **Missing Values (`NaN` / Blank)**:
   - Empty values across categorical columns (`Crop`, `State`) and numeric attributes (`Crop_Area_Hectares`, `Rainfall_mm`, `Market_Price_Per_Tonne`).
5. **Physically Impossible Values**:
   - Negative crop area (`-45.0 ha`).
   - Negative crop production (`-150.0 tonnes`).
   - Negative seasonal rainfall (`-35.0 mm`).
   - Unrealistic temperature readings (`115.0°C` and `-45.0°C`).
   - Zero or negative productivity yield (`0.00`, `-2.50`).
6. **Legitimate Agricultural Outliers**:
   - Large commercial sugarcane harvests (>30,000 tonnes).
   - Extreme flash flood rainfall (1850 mm).
   - Off-season wholesale pulse price spike (₹28,500 / tonne).

---

## 🛠️ Cleaning Methodology

### 1. Deduplication
Duplicates are identified using `.duplicated()` and pruned using `.drop_duplicates()`, preventing inflated sample counts.

### 2. String Normalization & Text Cleaning
Whitespace is removed using `.str.strip()`. Casing is standardized using `.str.title()`. Missing categorical values are filled using the modal class (`.mode()[0]`).

### 3. Type Conversion & Character Cleansing
Regex-free string replacement cleans characters:
```python
df[col] = df[col].astype(str).str.replace('₹', '').str.replace('$', '').str.replace(',', '').str.replace('ha', '')
df[col] = pd.to_numeric(df[col], errors='coerce')
```

### 4. Logical Bounds & Imputation
- Negative areas, productions, rainfalls, and out-of-range temperatures ($T < -10^\circ\text{C}$ or $T > 60^\circ\text{C}$) are replaced with `np.nan`.
- Missing values are imputed using the **Median**, which is resistant to skewness caused by extreme entries.

### 5. Derived Physical Metric Recalculation
Agricultural yield must satisfy:
$$\text{Yield} = \frac{\text{Production (Tonnes)}}{\text{Crop Area (Hectares)}}$$
All yield values are re-evaluated and rounded to two decimal places, removing discrepancies.

---

## 🌾 Outlier Treatment in Agribusiness

In generic software engineering or synthetic datasets, outliers are frequently dropped automatically. However, in **Agribusiness Analytics**, extreme data points often represent genuine real-world phenomena:

- **Sugarcane Production**: Sugarcane yields are significantly higher (~60–85 t/ha) than grains (~2–5 t/ha), resulting in production numbers exceeding 20,000 tonnes on standard plots.
- **Monsoon Floods**: Extreme localized downpours (e.g., 1850 mm) reflect genuine monsoon events.
- **Market Price Spikes**: Off-season shortages, crop failures, or niche cash crops frequently trigger 4x–5x price increases.

### IQR Detection Formula
$$\text{IQR} = Q_3 - Q_1$$
$$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$

Our pipeline detects and reports these values for domain assessment rather than deleting them, preserving ecological and economic insights.

---

## 💻 Technologies Used

- **Python 3.9+**: Core scripting language.
- **Pandas**: Tabular data manipulation, aggregation, and restructuring.
- **NumPy**: Numerical operations and NaN handling.
- **Matplotlib**: Generation of visual analytics summary charts.

---

## 📁 Project Structure

```text
agribusiness-data-cleaning/
│
├── data/
│   ├── raw_agriculture_data.csv          # Intentionally dirty input dataset (~104 rows)
│   ├── cleaned_agriculture_data.csv      # Cleaned, validated output dataset (100 rows)
│   └── cleaned_data_visual_summary.png   # Visual analytics chart generated by pipeline
│
├── src/
│   └── data_cleaning.py                  # Complete end-to-end data cleaning pipeline
│
├── README.md                             # Comprehensive project documentation
├── requirements.txt                      # Project dependencies
└── .gitignore                            # Standard Python ignore rules
```

---

## ⚙️ Installation & Setup

### Prerequisites
Make sure Python 3.9+ and Git are installed on your machine.

### Windows PowerShell Setup

```powershell
# 1. Clone or navigate to the repository folder
cd agribusiness-data-cleaning

# 2. Create a virtual environment named 'venv'
python -m venv venv

# 3. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# (Optional: If script execution is restricted in PowerShell, run once:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

Execute the cleaning pipeline from the project root:

```powershell
python src/data_cleaning.py
```

The script will:
1. Inspect raw data and display initial anomalies.
2. Execute all cleaning and imputation steps with progress logs.
3. Run validation checks.
4. Export the clean dataset to `data/cleaned_agriculture_data.csv`.
5. Generate a visual chart in `data/cleaned_data_visual_summary.png`.

---

## 🖥️ Example Output

```text
===========================================================================
      AGRIBUSINESS DATA COLLECTION AND CLEANING PIPELINE
===========================================================================

[INFO] Loading raw dataset from:
  -> data/raw_agriculture_data.csv

---------------------------------------------------------------------------
STEP 1: INITIAL DATA INSPECTION
---------------------------------------------------------------------------
Total Rows:    104
Total Columns: 9

Column Names in Dataset:
  - Date
  - State
  - Crop
  - Crop_Area_Hectares
  - Production_Tonnes
  - Yield_Tonnes_Per_Hectare
  - Rainfall_mm
  - Temperature_C
  - Market_Price_Per_Tonne

---------------------------------------------------------------------------
STEP 2: CHECKING MISSING VALUES & DUPLICATES
---------------------------------------------------------------------------
Missing Values Per Column (Before Cleaning):
State                     1
Crop                      1
Crop_Area_Hectares        1
Production_Tonnes         1
Rainfall_mm               2
Market_Price_Per_Tonne    1

Duplicate Records Found: 4

---------------------------------------------------------------------------
STEP 3: REMOVING DUPLICATE RECORDS
---------------------------------------------------------------------------
[SUCCESS] Removed 4 duplicate row(s). Remaining rows: 100

---------------------------------------------------------------------------
STEP 4: CLEANING TEXT COLUMNS (State & Crop)
---------------------------------------------------------------------------
[CLEANED] Text column 'State' trimmed of whitespace and standardized to Title Case.
[CLEANED] Text column 'Crop' trimmed of whitespace and standardized to Title Case.
[IMPUTED] Filled 1 missing value(s) in 'State' with Mode: 'Maharashtra'
[IMPUTED] Filled 1 missing value(s) in 'Crop' with Mode: 'Sugarcane'

---------------------------------------------------------------------------
STEP 5: CONVERTING NUMERICAL COLUMNS TO NUMERIC DATA TYPES
---------------------------------------------------------------------------
[CONVERTED] Column 'Crop_Area_Hectares' successfully parsed to Float64.
[CONVERTED] Column 'Production_Tonnes' successfully parsed to Float64.
[CONVERTED] Column 'Yield_Tonnes_Per_Hectare' successfully parsed to Float64.
[CONVERTED] Column 'Rainfall_mm' successfully parsed to Float64.
[CONVERTED] Column 'Temperature_C' successfully parsed to Float64.
[CONVERTED] Column 'Market_Price_Per_Tonne' successfully parsed to Float64.
[CONVERTED] Column 'Date' converted to datetime format (YYYY-MM-DD).

---------------------------------------------------------------------------
STEP 6: HANDLING INVALID VALUES & IMPUTING NUMERICAL DATA
---------------------------------------------------------------------------
[INVALID] Detected 1 record(s) with negative or zero Crop Area. Resetting to NaN.
[INVALID] Detected 1 record(s) with negative Production. Resetting to NaN.
[INVALID] Detected 1 record(s) with negative Rainfall. Resetting to NaN.
[INVALID] Detected 2 record(s) with unrealistic Temperature values. Resetting to NaN.
[IMPUTED] Column 'Crop_Area_Hectares': 2 missing value(s) filled with Median (205.00).
[IMPUTED] Column 'Production_Tonnes': 2 missing value(s) filled with Median (697.50).
[IMPUTED] Column 'Rainfall_mm': 3 missing value(s) filled with Median (70.00).
[IMPUTED] Column 'Temperature_C': 2 missing value(s) filled with Median (28.00).
[IMPUTED] Column 'Market_Price_Per_Tonne': 1 missing value(s) filled with Median (3220.00).

[VALIDATION] Recomputing 'Yield_Tonnes_Per_Hectare' using agricultural formula: (Production / Crop_Area)
[SUCCESS] All yield values recalculated and rounded to 2 decimal places.

---------------------------------------------------------------------------
STEP 7: OUTLIER DETECTION (IQR METHOD)
---------------------------------------------------------------------------
Column: Production_Tonnes
  - Normal Range: [-817.19, 2178.31]
  - Potential Outliers Count: 16 (e.g. Sugarcane high-biomass yields)

Column: Rainfall_mm
  - Normal Range: [-106.88, 258.12]
  - Potential Outliers Count: 1 (Monsoon flood event: 1850 mm)

Column: Market_Price_Per_Tonne
  - Normal Range: [-2188.12, 9736.88]
  - Potential Outliers Count: 1 (Off-season shortage spike: ₹28,500)

---------------------------------------------------------------------------
STEP 8: FINAL DATASET VALIDATION
---------------------------------------------------------------------------
Total Missing Values Remaining:       0
Total Duplicate Rows Remaining:       0
Negative Area Records:                0
Negative Production Records:          0
Negative Rainfall Records:            0
Unrealistic Temperature Records:      0

>>> [PASSED] ALL DATA VALIDATION CHECKS PASSED SUCCESSFULLY! <<<

---------------------------------------------------------------------------
STEP 9: BEFORE-AND-AFTER SUMMARY STATISTICS
---------------------------------------------------------------------------
[Before Cleaning - Raw Data Summary]:
  Shape: 104 rows x 9 columns
  Missing Values: 7
  Duplicates:     4

[After Cleaning - Clean Data Summary]:
  Shape: 100 rows x 9 columns
  Missing Values: 0
  Duplicates:     0
```

---

## 🔮 Future Improvements

1. **Automated Schema Validation**: Integrate `pydantic` or `pandera` to enforce strict schema types and value constraints at ingestion.
2. **Crop-Specific Median Imputation**: Group by crop category before imputing area and production for improved biological accuracy.
3. **Interactive Dashboard**: Build a Streamlit or Dash web application allowing agricultural officers to upload raw CSV files and view interactive cleaning dashboards.
4. **Geo-Spatial Verification**: Cross-reference state and coordinates with OpenStreetMap APIs or Indian meteorological boundaries.

---

## 📄 License

This project is licensed under the MIT License - feel free to use and adapt it for academic and research purposes.
