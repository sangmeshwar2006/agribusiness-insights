"""
================================================================================
Project: Data Collection and Cleaning for Agribusiness Analytics
File: data_cleaning.py
Author: Data Science Engineering Student
Description:
    A beginner-friendly data cleaning pipeline for agricultural datasets.
    This script inspects raw data, resolves formatting and data type issues,
    handles missing/duplicate values, identifies outliers via the IQR method,
    validates the clean dataset, and exports the clean data ready for visualization.
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# Use Agg backend for Matplotlib so it saves charts headlessly without popups
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure Windows PowerShell/CMD console handles UTF-8 characters like currency symbols
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 75)
    print("      AGRIBUSINESS DATA COLLECTION AND CLEANING PIPELINE")
    print("=" * 75)

    # --------------------------------------------------------------------------
    # STEP 1 & 2: SETUP FILE PATHS AND LOAD DATA
    # --------------------------------------------------------------------------
    # Determine base directory dynamically so the script runs from any folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    raw_data_path = os.path.join(project_root, "data", "raw_agriculture_data.csv")
    cleaned_data_path = os.path.join(project_root, "data", "cleaned_agriculture_data.csv")

    print(f"\n[INFO] Loading raw dataset from:\n  -> {raw_data_path}")
    
    # Load raw dataset into a Pandas DataFrame
    df_raw = pd.read_csv(raw_data_path)
    df = df_raw.copy()  # Work on a copy to preserve raw data

    # --------------------------------------------------------------------------
    # STEP 3: INITIAL DATA INSPECTION
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 1: INITIAL DATA INSPECTION")
    print("-" * 75)
    
    print(f"Total Rows:    {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nColumn Names in Dataset:")
    for col in df.columns:
        print(f"  - {col}")
        
    print("\nFirst 5 Records (Raw):")
    print(df.head().to_string(index=True))
    
    print("\nData Types (Raw):")
    print(df.dtypes.to_string())

    # --------------------------------------------------------------------------
    # STEP 4 & 5: CHECK MISSING VALUES AND DUPLICATES
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 2: CHECKING MISSING VALUES & DUPLICATES")
    print("-" * 75)
    
    raw_missing = df.isnull().sum()
    print("Missing Values Per Column (Before Cleaning):")
    print(raw_missing[raw_missing > 0].to_string() if raw_missing.sum() > 0 else "  No missing values found.")
    
    num_duplicates = df.duplicated().sum()
    print(f"\nDuplicate Records Found: {num_duplicates}")

    # --------------------------------------------------------------------------
    # STEP 6: REMOVE DUPLICATE RECORDS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 3: REMOVING DUPLICATE RECORDS")
    print("-" * 75)
    
    if num_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[SUCCESS] Removed {num_duplicates} duplicate row(s). Remaining rows: {len(df)}")
    else:
        print("[INFO] No duplicate rows to remove.")

    # --------------------------------------------------------------------------
    # STEP 7: CLEAN TEXT / CATEGORICAL COLUMNS
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 4: CLEANING TEXT COLUMNS (State & Crop)")
    print("-" * 75)
    
    # Text columns often contain irregular spaces and inconsistent casing (e.g. 'punjab', ' PUNJAB ')
    text_columns = ["State", "Crop"]
    for col in text_columns:
        # Strip leading and trailing whitespace
        df[col] = df[col].astype(str).str.strip()
        
        # Replace empty strings or string 'nan' with actual NaN
        df[col] = df[col].replace(["", "nan", "NaN", "null", "None"], np.nan)
        
        # Standardize capitalization to Title Case (e.g. 'Wheat', 'Punjab')
        df[col] = df[col].str.title()
        print(f"[CLEANED] Text column '{col}' trimmed of whitespace and standardized to Title Case.")

    # Handle missing values in categorical columns using the Mode (most frequent value)
    for col in text_columns:
        missing_cat_count = df[col].isnull().sum()
        if missing_cat_count > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"[IMPUTED] Filled {missing_cat_count} missing value(s) in '{col}' with Mode: '{mode_val}'")

    # --------------------------------------------------------------------------
    # STEP 8: CONVERT NUMERICAL COLUMNS TO APPROPRIATE DATA TYPES
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 5: CONVERTING NUMERICAL COLUMNS TO NUMERIC DATA TYPES")
    print("-" * 75)
    
    numeric_columns = [
        "Crop_Area_Hectares",
        "Production_Tonnes",
        "Yield_Tonnes_Per_Hectare",
        "Rainfall_mm",
        "Temperature_C",
        "Market_Price_Per_Tonne"
    ]
    
    for col in numeric_columns:
        # Convert to string to clean extraneous characters (e.g., currency symbols, commas, units like 'ha')
        df[col] = df[col].astype(str).str.replace("₹", "", regex=False)
        df[col] = df[col].str.replace("$", "", regex=False)
        df[col] = df[col].str.replace(",", "", regex=False)
        df[col] = df[col].str.replace("ha", "", regex=False)
        df[col] = df[col].str.replace("mm", "", regex=False)
        df[col] = df[col].str.strip()
        
        # Convert to float; invalid strings become NaN (Not a Number)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"[CONVERTED] Column '{col}' successfully parsed to Float64.")

    # Convert Date column to standard datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    print("[CONVERTED] Column 'Date' converted to datetime format (YYYY-MM-DD).")

    # --------------------------------------------------------------------------
    # STEP 9, 10 & 11: HANDLE INVALID VALUES AND MISSING NUMERICAL DATA
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 6: HANDLING INVALID VALUES & IMPUTING NUMERICAL DATA")
    print("-" * 75)

    # 1. Negative Crop Area is physically impossible -> Flag as NaN
    neg_area = (df["Crop_Area_Hectares"] <= 0)
    if neg_area.sum() > 0:
        print(f"[INVALID] Detected {neg_area.sum()} record(s) with negative or zero Crop Area. Resetting to NaN.")
        df.loc[neg_area, "Crop_Area_Hectares"] = np.nan

    # 2. Negative Production is physically impossible -> Flag as NaN
    neg_prod = (df["Production_Tonnes"] < 0)
    if neg_prod.sum() > 0:
        print(f"[INVALID] Detected {neg_prod.sum()} record(s) with negative Production. Resetting to NaN.")
        df.loc[neg_prod, "Production_Tonnes"] = np.nan

    # 3. Negative Rainfall is physically impossible -> Flag as NaN
    neg_rain = (df["Rainfall_mm"] < 0)
    if neg_rain.sum() > 0:
        print(f"[INVALID] Detected {neg_rain.sum()} record(s) with negative Rainfall. Resetting to NaN.")
        df.loc[neg_rain, "Rainfall_mm"] = np.nan

    # 4. Unrealistic Temperature (e.g. > 60°C or < -10°C for Indian agricultural lands) -> Flag as NaN
    unrealistic_temp = (df["Temperature_C"] > 60.0) | (df["Temperature_C"] < -10.0)
    if unrealistic_temp.sum() > 0:
        print(f"[INVALID] Detected {unrealistic_temp.sum()} record(s) with unrealistic Temperature values. Resetting to NaN.")
        df.loc[unrealistic_temp, "Temperature_C"] = np.nan

    # Impute missing numerical values using Median (median is resilient to extreme values)
    imputation_cols = ["Crop_Area_Hectares", "Production_Tonnes", "Rainfall_mm", "Temperature_C", "Market_Price_Per_Tonne"]
    for col in imputation_cols:
        col_median = df[col].median()
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna(col_median)
            print(f"[IMPUTED] Column '{col}': {missing_count} missing value(s) filled with Median ({col_median:.2f}).")

    # 5. Domain-Specific Calculation: Yield = Production_Tonnes / Crop_Area_Hectares
    # Check for invalid yield (<= 0 or missing or mathematically inconsistent)
    print("\n[VALIDATION] Recomputing 'Yield_Tonnes_Per_Hectare' using agricultural formula: (Production / Crop_Area)")
    df["Yield_Tonnes_Per_Hectare"] = (df["Production_Tonnes"] / df["Crop_Area_Hectares"]).round(2)
    print("[SUCCESS] All yield values recalculated and rounded to 2 decimal places.")

    # --------------------------------------------------------------------------
    # STEP 12 & 13: OUTLIER DETECTION USING THE IQR (INTERQUARTILE RANGE) METHOD
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 7: OUTLIER DETECTION (IQR METHOD)")
    print("-" * 75)
    print("Note: In agriculture, extreme values (e.g. bumper harvests, localized drought,")
    print("or market price spikes) often represent genuine physical events rather than errors.")
    print("Therefore, we detect and flag outliers for business review without blind deletion.\n")

    check_outlier_cols = ["Production_Tonnes", "Rainfall_mm", "Market_Price_Per_Tonne"]
    
    for col in check_outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"Column: {col}")
        print(f"  - Q1 (25%): {q1:.2f}, Q3 (75%): {q3:.2f}, IQR: {iqr:.2f}")
        print(f"  - Normal Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  - Potential Outliers Count: {len(outliers)}")
        if len(outliers) > 0:
            sample_vals = outliers[col].values[:3]
            print(f"  - Sample Outlier Values: {sample_vals}")
        print()

    # --------------------------------------------------------------------------
    # STEP 14: VALIDATION OF THE CLEANED DATASET
    # --------------------------------------------------------------------------
    print("-" * 75)
    print("STEP 8: FINAL DATASET VALIDATION")
    print("-" * 75)
    
    clean_missing = df.isnull().sum().sum()
    clean_duplicates = df.duplicated().sum()
    neg_area_clean = (df["Crop_Area_Hectares"] <= 0).sum()
    neg_prod_clean = (df["Production_Tonnes"] < 0).sum()
    neg_rain_clean = (df["Rainfall_mm"] < 0).sum()
    invalid_temp_clean = ((df["Temperature_C"] > 60) | (df["Temperature_C"] < -10)).sum()
    
    print(f"Total Missing Values Remaining:       {clean_missing}")
    print(f"Total Duplicate Rows Remaining:       {clean_duplicates}")
    print(f"Negative Area Records:                {neg_area_clean}")
    print(f"Negative Production Records:          {neg_prod_clean}")
    print(f"Negative Rainfall Records:            {neg_rain_clean}")
    print(f"Unrealistic Temperature Records:      {invalid_temp_clean}")

    if (clean_missing == 0 and clean_duplicates == 0 and 
        neg_area_clean == 0 and neg_prod_clean == 0 and 
        neg_rain_clean == 0 and invalid_temp_clean == 0):
        print("\n>>> [PASSED] ALL DATA VALIDATION CHECKS PASSED SUCCESSFULLY! <<<")
    else:
        print("\n>>> [WARNING] Some validation issues remain. Please inspect. <<<")

    # --------------------------------------------------------------------------
    # STEP 15: BEFORE-AND-AFTER COMPARISON
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 9: BEFORE-AND-AFTER SUMMARY STATISTICS")
    print("-" * 75)
    
    print("\n[Before Cleaning - Raw Data Summary]:")
    print(f"  Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    print(f"  Missing Values: {df_raw.isnull().sum().sum()}")
    print(f"  Duplicates:     {df_raw.duplicated().sum()}")
    
    print("\n[After Cleaning - Clean Data Summary]:")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Missing Values: {df.isnull().sum().sum()}")
    print(f"  Duplicates:     {df.duplicated().sum()}")

    print("\nKey Numerical Summary (Cleaned Dataset):")
    summary_stats = df[["Crop_Area_Hectares", "Production_Tonnes", "Yield_Tonnes_Per_Hectare", "Rainfall_mm", "Temperature_C", "Market_Price_Per_Tonne"]].describe().round(2)
    print(summary_stats.to_string())

    # --------------------------------------------------------------------------
    # STEP 16: SAVE THE CLEANED DATASET
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 10: SAVING CLEANED DATASET")
    print("-" * 75)
    
    # Format Date column back to YYYY-MM-DD string for clean CSV output
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(cleaned_data_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved to:\n  -> {cleaned_data_path}")

    # --------------------------------------------------------------------------
    # STEP 11: GENERATE VISUALIZATION (READY FOR AGRIBUSINESS ANALYTICS)
    # --------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STEP 11: GENERATING VISUALIZATION (READY FOR AGRIBUSINESS ANALYTICS)")
    print("-" * 75)
    try:
        plot_path = os.path.join(project_root, "data", "cleaned_data_visual_summary.png")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Average Yield by Crop
        avg_yield = df.groupby("Crop")["Yield_Tonnes_Per_Hectare"].mean().sort_values(ascending=False)
        avg_yield.plot(kind="bar", ax=axes[0], color="#2E7D32", edgecolor="black")
        axes[0].set_title("Average Yield by Crop (Tonnes / Hectare)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Crop", fontsize=10)
        axes[0].set_ylabel("Yield (Tonnes/Ha)", fontsize=10)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)

        # Plot 2: Average Market Price by Crop
        avg_price = df.groupby("Crop")["Market_Price_Per_Tonne"].mean().sort_values(ascending=False)
        avg_price.plot(kind="bar", ax=axes[1], color="#E65100", edgecolor="black")
        axes[1].set_title("Average Market Price by Crop (INR / Tonne)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Crop", fontsize=10)
        axes[1].set_ylabel("Price (INR/Tonne)", fontsize=10)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[SUCCESS] Visual analytics chart generated:\n  -> {plot_path}")
    except Exception as e:
        print(f"[INFO] Skipping chart generation: {e}")

    # --------------------------------------------------------------------------
    # STEP 12: FINAL CLEANING SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("                 PIPELINE EXECUTION COMPLETE SUMMARY")
    print("=" * 75)
    print(" 1. Duplicate rows identified and purged.")
    print(" 2. Text fields (State, Crop) trimmed and title-cased.")
    print(" 3. Categorical missing values filled with mode.")
    print(" 4. Numeric fields parsed, cleaned of currency signs and text symbols.")
    print(" 5. Impossible negative and extreme values flagged and imputed with median.")
    print(" 6. Crop Yield mathematically recomputed: (Production / Crop_Area).")
    print(" 7. Outliers identified via IQR method with domain explanations.")
    print(" 8. Clean dataset fully validated and exported for visualization.")
    print(" 9. Generated agricultural analytics visual summary chart.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
