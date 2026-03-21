"""
Clean and preprocess Chicago crime data.

This module implements the full data cleaning pipeline: dropping redundant
columns, handling missing values, parsing dates and adding temporal features,
adding spatial bins (Lat_bin, Lon_bin) and crime categories, deduplication,
type conversion, and filtering out coordinates outside Chicago bounds.

Output is saved to data/processed/crimes_cleaned.csv for use by EDA and modeling.

Source: Extracted from notebooks/02_data_cleaning.ipynb
"""

from pathlib import Path

import pandas as pd

from .load_data import DATA_RAW, DATA_PROCESSED, load_raw_crimes

# ------------------------------------------------------------------------------
# Crime category mapping
# ------------------------------------------------------------------------------
# Groups Police Primary Type into 5 categories for modeling. Aligns with
# prediction task in notebooks/04_data_modeling.ipynb.
CRIME_CATEGORY_MAP = {
    "Violent": ["BATTERY", "ASSAULT", "ROBBERY", "CRIMINAL SEXUAL ASSAULT", "HOMICIDE", "KIDNAPPING"],
    "Theft": ["THEFT", "MOTOR VEHICLE THEFT", "BURGLARY"],
    "Property": ["CRIMINAL DAMAGE", "CRIMINAL TRESPASS", "ARSON"],
    "Narcotics": ["NARCOTICS", "OTHER NARCOTIC VIOLATION"],
    "Other": [],  # Default for all unlisted types (e.g., DECEPTIVE PRACTICE, GAMBLING)
}

# Chicago bounding box (approx) - filter out erroneous coordinates
# Lat 41.6-42.1, Lon -87.95 to -87.5 covers the city
CHICAGO_LAT = (41.6, 42.1)
CHICAGO_LON = (-87.95, -87.5)


def get_hour_bin(h: int) -> int:
    """
    Map hour (0-23) to 6-bin time-of-day for temporal modeling.

    Bins: 0=0-4, 1=4-8, 2=8-12, 3=12-16, 4=16-20, 5=20-24.
    Used for HourBin feature in crime type and beat prediction.
    """
    if 0 <= h < 4:
        return 0
    if 4 <= h < 8:
        return 1
    if 8 <= h < 12:
        return 2
    if 12 <= h < 16:
        return 3
    if 16 <= h < 20:
        return 4
    return 5


def map_crime_category(pt: str) -> str:
    """
    Map Police Primary Type (e.g., 'BATTERY', 'THEFT') to 5-category Crime_Category.

    Unknown types default to 'Other'. Used for both EDA and as a modeling target.
    """
    for cat, types in CRIME_CATEGORY_MAP.items():
        if pt in types:
            return cat
    return "Other"


def clean_data(df: pd.DataFrame | None = None, raw_path: Path | str | None = None) -> pd.DataFrame:
    """
    Run full cleaning pipeline on raw Chicago crime data.

    Pipeline steps:
        1. Drop redundant columns (Case Number, Location, X/Y Coordinate, Updated On)
        2. Handle missing: drop rows with missing Lat/Lon, fill Location/Community Area
        3. Parse Date and add Year, Month, DayOfWeek, Hour, Quarter, WeekOfYear, IsWeekend
        4. Add Lat_bin, Lon_bin (0.02 deg grid ~2km), HourBin, Crime_Category
        5. Drop duplicate rows by ID
        6. Convert Arrest, Domestic, Ward to numeric
        7. Filter rows with coordinates outside Chicago bounds

    Args:
        df: Optional DataFrame to clean. If None, loads from raw_path or default.
        raw_path: Path to raw CSV when df is None.

    Returns:
        Cleaned DataFrame ready for EDA and modeling.
    """
    if df is None:
        df = load_raw_crimes(raw_path or (DATA_RAW / "crimes.csv"))
    df = df.copy()

    # Step 1: Drop redundant columns
    cols_to_drop = ["Case Number", "Location", "X Coordinate", "Y Coordinate", "Updated On"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Step 2: Handle missing values
    # Lat/Lon required for spatial analysis; drop rows without them
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Location Description"] = df["Location Description"].fillna("UNKNOWN")
    df["Community Area"] = df["Community Area"].fillna(-1).astype(int)

    # Step 3: Parse Date and add temporal features
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["DayOfWeekName"] = df["Date"].dt.day_name()
    df["Hour"] = df["Date"].dt.hour
    df["Quarter"] = df["Date"].dt.quarter
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6])  # Saturday, Sunday

    # Step 3b: Add spatial bins and Crime_Category for modeling
    # 0.02 deg ~ 2km for Chicago; enables spatial aggregation
    df["Lat_bin"] = (df["Latitude"] // 0.02).astype(int)
    df["Lon_bin"] = (df["Longitude"] // 0.02).astype(int)
    df["HourBin"] = df["Hour"].apply(get_hour_bin)
    df["Crime_Category"] = df["Primary Type"].apply(map_crime_category)

    # Step 4: Handle duplicates
    df = df.drop_duplicates(subset=["ID"])

    # Step 5: Ensure correct data types for ML
    df["Arrest"] = df["Arrest"].astype(int)
    df["Domestic"] = df["Domestic"].astype(int)
    df["Ward"] = df["Ward"].fillna(-1).astype(int)

    # Step 6: Filter coordinates outside Chicago (data quality)
    mask = df["Latitude"].between(*CHICAGO_LAT) & df["Longitude"].between(*CHICAGO_LON)
    df = df[mask]

    return df


def save_cleaned(df: pd.DataFrame, out_path: Path | str | None = None) -> Path:
    """
    Save cleaned DataFrame to CSV.

    Creates parent directory if it does not exist.

    Args:
        df: Cleaned DataFrame from clean_data().
        out_path: Output path. Default: data/processed/crimes_cleaned.csv.

    Returns:
        Path where file was saved.
    """
    out_path = out_path or (DATA_PROCESSED / "crimes_cleaned.csv")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return Path(out_path)


def load_cleaned(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Load the pre-cleaned crimes CSV (output of clean_data + save_cleaned).

    Use this in EDA and modeling to skip the cleaning step.
    """
    path = filepath or (DATA_PROCESSED / "crimes_cleaned.csv")
    return pd.read_csv(path)


if __name__ == "__main__":
    # Run full clean-and-save when executed as a script
    df = load_raw_crimes()
    print("Initial shape:", df.shape)
    df = clean_data(df)
    print("After cleaning:", df.shape)
    out = save_cleaned(df)
    print(f"Saved to {out}")
