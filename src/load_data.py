"""
Load and inspect Chicago crime data.

This module provides functions to load the raw Chicago Police Department crime dataset
and perform initial data understanding: structure overview, missing value analysis,
primary crime type distribution, geographic and temporal column inspection.

Source: Extracted from notebooks/01_data_understanding.ipynb
Data: Chicago Police Department - City of Chicago Data Portal
      (https://data.cityofchicago.org/)
"""

from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------------------
# Path configuration
# ------------------------------------------------------------------------------
# All paths are relative to the project root so the package works regardless of
# the current working directory when imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"  # Raw CSV before cleaning
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"  # Cleaned output from preprocess.py


def load_raw_crimes(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Load the Chicago crimes dataset from raw CSV.

    Args:
        filepath: Optional path to the CSV file. If None, uses
                  data/raw/crimes.csv relative to project root.

    Returns:
        DataFrame with raw crime records. Expected columns include ID, Date,
        Primary Type, Location Description, Beat, Latitude, Longitude, etc.
    """
    path = filepath or (DATA_RAW / "crimes.csv")
    return pd.read_csv(path)


def get_dataset_overview(df: pd.DataFrame) -> None:
    """
    Print dataset shape, column names/types, and basic memory stats.

    Useful for initial data understanding before cleaning or modeling.
    """
    print("Dataset shape:", df.shape)
    print("\nColumn names and types:")
    print(df.dtypes)
    print("\nBasic stats:")
    df.info()


def get_missing_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing value counts and percentages for all columns.

    Only returns rows where at least one value is missing. Sorted by
    missing count descending.

    Returns:
        DataFrame with columns 'Missing Count' and 'Missing %'.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    return missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)


def get_primary_type_stats(df: pd.DataFrame) -> None:
    """
    Print distribution of Primary Type (crime category from police classification).

    Primary Type is a key target for prediction (e.g., Theft, Battery, Assault).
    """
    primary_counts = df["Primary Type"].value_counts()
    print("Primary Type - Top 15:")
    print(primary_counts.head(15).to_string())
    print("\nTotal unique primary types:", df["Primary Type"].nunique())


def get_location_stats(df: pd.DataFrame) -> None:
    """
    Print distribution of Location Description (where crime occurred).

    Examples: STREET, RESIDENCE, SCHOOL, etc. Relevant for spatial modeling.
    """
    loc_counts = df["Location Description"].value_counts(dropna=False)
    print("Location Description - Top 20:")
    print(loc_counts.head(20).to_string())
    print("\nTotal unique location types:", df["Location Description"].nunique())


def get_geographic_stats(df: pd.DataFrame) -> None:
    """
    Print stats for geographic columns used in location-based predictions.

    Beat is the primary unit for police deployment (patrol areas). District,
    Ward, and Community Area provide coarser geography. Lat/Lon enable
    spatial binning and mapping.
    """
    print("District - unique count:", df["District"].nunique(), "| Sample:", sorted(df["District"].dropna().unique())[:10])
    print("Ward - unique count:", df["Ward"].nunique(), "| Range:", df["Ward"].min(), "-", df["Ward"].max())
    print("Community Area - unique count:", df["Community Area"].nunique(), "| Missing:", df["Community Area"].isnull().sum())
    print("Beat - unique count:", df["Beat"].nunique())
    print("\nLatitude/Longitude coverage:", df["Latitude"].notna().sum(), "/", len(df))


def verify_date_parsing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify that the Date column can be parsed with the expected format.

    Raw data uses '%m/%d/%Y %I:%M:%S %p' (e.g., "03/13/2026 12:00:00 AM").
    Returns a temporary DataFrame with parsed dates; useful for debugging.

    Returns:
        Copy of df with added 'Date_parsed' column.
    """
    df_temp = df.copy()
    df_temp["Date_parsed"] = pd.to_datetime(df_temp["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    print("Sample parsed dates:")
    print(df_temp[["Date", "Date_parsed"]].dropna().head())
    print("\nParse errors (null):", df_temp["Date_parsed"].isnull().sum())
    return df_temp


def get_categorical_stats(df: pd.DataFrame) -> None:
    """
    Print distributions for Arrest (binary), Domestic (binary), and FBI Code.

    FBI Code is a standardized crime classification used for federal reporting.
    """
    print("Arrest distribution:", df["Arrest"].value_counts().to_dict())
    print("\nDomestic distribution:", df["Domestic"].value_counts().to_dict())
    print("\nFBI Code - top 10:", df["FBI Code"].value_counts().head(10).to_string())


def get_duplicate_stats(df: pd.DataFrame) -> None:
    """
    Check for duplicate rows and redundant columns.

    Location is (Latitude, Longitude) tuple - redundant with Lat/Lon columns.
    ID should be unique; Case Number may have duplicates.
    """
    print("Location sample:", df["Location"].iloc[0])
    print("Latitude, Longitude:", df["Latitude"].iloc[0], df["Longitude"].iloc[0])
    print("\nDuplicate rows (by ID):", df.duplicated(subset=["ID"]).sum())
    print("Duplicate Case Numbers:", df["Case Number"].duplicated().sum())


if __name__ == "__main__":
    # Run basic data understanding when executed as a script
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", "{:.2f}".format)

    df = load_raw_crimes()
    print("Loaded", len(df), "rows")
    get_dataset_overview(df)
    print("\nMissing values:")
    print(get_missing_analysis(df))
