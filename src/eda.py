"""
Exploratory data analysis for Chicago crime data.

This module provides visualization functions to explore crime patterns:
- Geographic: top beats, community areas, density hexbin
- Crime type: primary type distribution, location types
- Temporal: crime by hour, day of week, month
- Cross-tabulation: crime type vs hour heatmap, top crimes by beat
- Binary outcomes: arrest rate, domestic incidents, weekday vs weekend

All plots use matplotlib and display via plt.show(). Run run_all_eda() to
execute the full suite, or call individual plot functions with a DataFrame.

Source: Extracted from notebooks/03_exploratory_data_analysis.ipynb
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .load_data import DATA_PROCESSED
from .preprocess import load_cleaned


def load_for_eda(filepath: Path | str | None = None) -> pd.DataFrame:
    """
    Load cleaned data and ensure Date is parsed as datetime.

    Required for EDA functions that use date-based grouping or plotting.
    """
    df = load_cleaned(filepath or (DATA_PROCESSED / "crimes_cleaned.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ------------------------------------------------------------------------------
# Geographic visualizations
# ------------------------------------------------------------------------------

def plot_top_beats(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Horizontal bar chart of top N beats by crime count.

    Beat is the primary unit for police deployment; this shows hot spots
    where patrol presence may be most needed.
    """
    beat_counts = df["Beat"].value_counts().sort_values(ascending=True)
    beat_counts.tail(top_n).plot(kind="barh", figsize=(10, 7), title=f"Top {top_n} Beats by Crime Count")
    plt.xlabel("Crime Count")
    plt.tight_layout()
    plt.show()


def plot_community_areas(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Horizontal bar chart of top N community areas by crime count.

    Community Area >= 0 filters out rows where Community Area was unknown (-1).
    """
    comm = df[df["Community Area"] >= 0]["Community Area"].value_counts().head(top_n)
    comm.plot(kind="barh", figsize=(10, 5), title=f"Top {top_n} Community Areas by Crime Count")
    plt.xlabel("Crime Count")
    plt.tight_layout()
    plt.show()


def plot_crime_density_hexbin(df: pd.DataFrame, sample_size: int = 50000) -> None:
    """
    Hexbin density map of crime locations (longitude vs latitude).

    Samples the dataset if large to keep rendering fast. Color intensity
    indicates crime count per hex; useful for identifying hot zones.
    """
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    plt.figure(figsize=(12, 10))
    plt.hexbin(sample["Longitude"], sample["Latitude"], gridsize=60, cmap="YlOrRd", mincnt=1)
    plt.colorbar(label="Crime count")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Crime Density - Chicago (hexbin)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Crime type and location visualizations
# ------------------------------------------------------------------------------

def plot_primary_type_distribution(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Side-by-side bar and pie chart of primary crime types.

    Bar: counts for top N types. Pie: percentage share of top 10.
    """
    type_counts = df["Primary Type"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    type_counts.head(top_n).plot(kind="barh", ax=axes[0], title=f"Top {top_n} Crime Types")
    axes[0].set_xlabel("Count")
    type_counts.head(10).plot(kind="pie", ax=axes[1], autopct="%1.1f%%", title="Top 10 Crime Types (%)")
    plt.tight_layout()
    plt.show()


def plot_location_types(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Horizontal bar chart of top N location types (e.g., STREET, RESIDENCE).

    Shows where crimes most frequently occur.
    """
    loc_counts = df["Location Description"].value_counts().head(top_n)
    loc_counts.plot(kind="barh", figsize=(10, 6), title=f"Top {top_n} Location Types for Crime")
    plt.xlabel("Crime Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Temporal visualizations
# ------------------------------------------------------------------------------

def plot_crime_by_hour(df: pd.DataFrame) -> None:
    """
    Bar chart of crime count by hour of day (0-23).

    Reveals diurnal patterns (e.g., peaks in evening, troughs in early morning).
    """
    hour_counts = df["Hour"].value_counts().sort_index()
    hour_counts.plot(kind="bar", figsize=(12, 5), title="Crime Count by Hour of Day")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_crime_by_day_of_week(df: pd.DataFrame) -> None:
    """
    Bar chart of crime count by day of week (Monday-Sunday).

    DayOfWeekName must exist (added in preprocess).
    """
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = df["DayOfWeekName"].value_counts().reindex(dow_order)
    dow_counts.plot(kind="bar", figsize=(10, 5), title="Crime Count by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_crime_by_month(df: pd.DataFrame) -> None:
    """
    Stacked/grouped bar chart of crime count by month, split by year.

    Helps identify seasonality and year-over-year trends.
    """
    month_counts = df.groupby(["Year", "Month"]).size().unstack(fill_value=0)
    month_counts.T.plot(kind="bar", figsize=(12, 5), title="Crime Count by Month (by Year)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.legend(title="Year")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Cross-tabulation and heatmaps
# ------------------------------------------------------------------------------

def plot_crime_type_vs_hour_heatmap(df: pd.DataFrame, top_types: int = 8) -> None:
    """
    Heatmap: crime type (rows) vs hour of day (columns).

    Shows which crime types peak at which hours. Uses top N crime types
    to keep the heatmap readable.
    """
    top_types_list = df["Primary Type"].value_counts().head(top_types).index
    top_df = df[df["Primary Type"].isin(top_types_list)]
    heat_data = top_df.pivot_table(
        index="Primary Type", columns="Hour", values="ID", aggfunc="count", fill_value=0
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(heat_data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(label="Count")
    plt.yticks(range(len(heat_data.index)), heat_data.index)
    plt.xticks(range(24), range(24))
    plt.xlabel("Hour")
    plt.ylabel("Primary Type")
    plt.title("Crime Type vs Hour of Day (Heatmap)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------------
# Binary and aggregated visualizations
# ------------------------------------------------------------------------------

def plot_arrest_domestic(df: pd.DataFrame) -> None:
    """
    Pie charts for Arrest and Domestic distributions.

    Arrest: whether an arrest was made. Domestic: whether incident was domestic.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["Arrest"].value_counts().plot(kind="pie", ax=axes[0], autopct="%1.1f%%", labels=["No Arrest", "Arrest"])
    df["Domestic"].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.1f%%", labels=["Non-Domestic", "Domestic"])
    axes[0].set_title("Arrest Rate")
    axes[1].set_title("Domestic Incidents")
    plt.tight_layout()
    plt.show()


def plot_top_crimes_by_beat(df: pd.DataFrame, top_beats: int = 8) -> None:
    """
    Grouped bar chart: crime type (x-axis) by beat (grouped bars).

    Shows which crime types dominate in the highest-crime beats. Useful
    for understanding geographic-crime type relationships.
    """
    top_beats_list = df["Beat"].value_counts().head(top_beats).index
    beat_type = df[df["Beat"].isin(top_beats_list)].groupby(["Beat", "Primary Type"]).size().unstack(fill_value=0)
    beat_type.loc[top_beats_list].T.plot(
        kind="bar", stacked=False, figsize=(14, 6), title=f"Top Crime Types in Top {top_beats} Beats"
    )
    plt.xlabel("Primary Type")
    plt.ylabel("Count")
    plt.legend(title="Beat")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_weekday_vs_weekend(df: pd.DataFrame) -> None:
    """
    Bar chart comparing crime count on weekdays vs weekend.

    IsWeekend is True for Saturday (5) and Sunday (6).
    """
    wk = df.groupby("IsWeekend").size()
    wk.index = ["Weekday", "Weekend"]
    wk.plot(kind="bar", figsize=(6, 4), title="Crime: Weekday vs Weekend")
    plt.xlabel("")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def run_all_eda(
    df: pd.DataFrame | None = None,
    filepath: Path | str | None = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    """
    Run the full EDA suite: all 12 plot functions in sequence.

    Args:
        df: Optional DataFrame. If None, loads from filepath or default.
        filepath: Path to cleaned CSV when df is None.
        show_plots: If False, disables plt.show() (e.g., for batch/CI runs).

    Returns:
        The DataFrame used (either passed in or loaded).
    """
    if df is None:
        df = load_for_eda(filepath)
    print("Shape:", df.shape)
    if not show_plots:
        plt.ioff()
    plot_top_beats(df)
    plot_community_areas(df)
    plot_crime_density_hexbin(df)
    plot_primary_type_distribution(df)
    plot_location_types(df)
    plot_crime_by_hour(df)
    plot_crime_by_day_of_week(df)
    plot_crime_by_month(df)
    plot_crime_type_vs_hour_heatmap(df)
    plot_arrest_domestic(df)
    plot_top_crimes_by_beat(df)
    plot_weekday_vs_weekend(df)
    return df


if __name__ == "__main__":
    df = run_all_eda()
