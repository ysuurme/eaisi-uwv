"""
Polars Join Patterns for Table 80072ned
Dataset: Ziekteverzuimpercentage (Sick Leave %)
"""

import polars as pl
from sqlalchemy import create_engine
from pathlib import Path

# Helper function to read any table from bronze database
def read_bronze_table(table_name: str) -> pl.DataFrame:
    """Read a table from the bronze database."""
    return pl.read_database(
        query=f'SELECT * FROM "{table_name}"',
        connection=get_engine()
    )

# Helper function to get table info
def get_table_info(table_name: str):
    """Display basic information about a table."""
    df = read_bronze_table(table_name)
    print(f"Table: {table_name}")
    print(f"Shape: {df.shape} (rows × columns)")
    print(f"\nColumns: {df.columns}")
    print(f"\nSchema:")
    print(df.schema)
    print(f"\nTop 5 rows:")
    print(df.head(5))
    return df



def get_engine():
    """Create database connection engine."""
    # Get path relative to this file's location
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    db_path = project_root / "data" / "1_bronze" / "bronze_data.db"
    return create_engine(f"sqlite:///{db_path}")


def load_80072_tables():
    """Load all 80072ned tables into Polars DataFrames."""
    engine = get_engine()

    tables = {
        "fact": pl.read_database('SELECT * FROM "80072ned_fact"', connection=engine),
        "dim_perioden": pl.read_database('SELECT * FROM "80072ned_dim_Perioden"', connection=engine),
        "dim_bedrijfskenmerken": pl.read_database('SELECT * FROM "80072ned_dim_BedrijfskenmerkenSBI2008"', connection=engine),
        "dim_category_groups": pl.read_database('SELECT * FROM "80072ned_dim_CategoryGroups"', connection=engine),
        "dim_data_properties": pl.read_database('SELECT * FROM "80072ned_dim_DataProperties"', connection=engine),
        "dim_table_infos": pl.read_database('SELECT * FROM "80072ned_dim_TableInfos"', connection=engine),
    }

    return tables


def basic_join():
    """Basic join of fact table with all dimensions."""
    tables = load_80072_tables()

    result = (
        tables["fact"]
        # Join with Period dimension
        .join(
            tables["dim_perioden"],
            left_on="Perioden",
            right_on="Key",
            how="left",
            suffix="_period"
        )
        # Join with Business Sector dimension
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        # Join with Category Groups
        .join(
            tables["dim_category_groups"],
            left_on="CategoryGroupID",
            right_on="ID",
            how="left",
            suffix="_category"
        )
        # Select and rename columns for clarity
        .select([
            pl.col("Perioden").alias("period_code"),
            pl.col("Title_period").alias("period_name"),
            pl.col("Status").alias("period_status"),
            pl.col("BedrijfskenmerkenSBI2008").alias("sector_code"),
            pl.col("Title_sector").alias("sector_name"),
            pl.col("Title_category").alias("category_group"),
            pl.col("Ziekteverzuimpercentage_1").alias("sick_leave_pct"),
        ])
        # Filter out nulls
        .filter(pl.col("sick_leave_pct").is_not_null())
        # Sort by period (descending) and sector name
        .sort(["period_code", "sector_name"], descending=[True, False])
    )

    return result


def time_series_by_sector(sector_code: str = "T001081"):
    """
    Get time series for a specific sector.
    Default: T001081 = All economic activities
    """
    tables = load_80072_tables()

    result = (
        tables["fact"]
        .join(
            tables["dim_perioden"],
            left_on="Perioden",
            right_on="Key",
            how="left",
            suffix="_period"
        )
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        .filter(pl.col("BedrijfskenmerkenSBI2008") == sector_code)
        .filter(pl.col("Ziekteverzuimpercentage_1").is_not_null())
        .select([
            pl.col("Perioden").alias("period_code"),
            pl.col("Title_period").alias("period_name"),
            pl.col("Title_sector").alias("sector_name"),
            pl.col("Ziekteverzuimpercentage_1").alias("sick_leave_pct"),
        ])
        .sort("period_code")
    )

    return result


def sector_comparison_latest():
    """Compare all sectors in the latest quarter."""
    tables = load_80072_tables()

    # Get latest period
    latest_period = tables["dim_perioden"].select("Key").sort("Key", descending=True).head(1).item()

    result = (
        tables["fact"]
        .filter(pl.col("Perioden") == latest_period)
        .filter(pl.col("Ziekteverzuimpercentage_1").is_not_null())
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        .join(
            tables["dim_category_groups"],
            left_on="CategoryGroupID",
            right_on="ID",
            how="left",
            suffix="_category"
        )
        .select([
            pl.lit(latest_period).alias("period"),
            pl.col("Title_sector").alias("sector_name"),
            pl.col("Title_category").alias("category_group"),
            pl.col("Ziekteverzuimpercentage_1").alias("sick_leave_pct"),
        ])
        .sort("sick_leave_pct", descending=True)
    )

    return result


def aggregate_by_category():
    """Aggregate sick leave percentage by category group."""
    tables = load_80072_tables()

    result = (
        tables["fact"]
        .filter(pl.col("Ziekteverzuimpercentage_1").is_not_null())
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        .join(
            tables["dim_category_groups"],
            left_on="CategoryGroupID",
            right_on="ID",
            how="left",
            suffix="_category"
        )
        .filter(pl.col("Title_category").is_not_null())
        .group_by("Title_category")
        .agg([
            pl.count().alias("record_count"),
            pl.col("Ziekteverzuimpercentage_1").mean().round(2).alias("avg_sick_leave_pct"),
            pl.col("Ziekteverzuimpercentage_1").min().round(2).alias("min_sick_leave_pct"),
            pl.col("Ziekteverzuimpercentage_1").max().round(2).alias("max_sick_leave_pct"),
        ])
        .sort("avg_sick_leave_pct", descending=True)
    )

    return result


def year_over_year_comparison(sector_code: str = "T001081"):
    """
    Calculate average sick leave by year for a specific sector.
    Default: T001081 = All economic activities
    """
    tables = load_80072_tables()

    result = (
        tables["fact"]
        .filter(pl.col("BedrijfskenmerkenSBI2008") == sector_code)
        .filter(pl.col("Ziekteverzuimpercentage_1").is_not_null())
        .join(
            tables["dim_perioden"],
            left_on="Perioden",
            right_on="Key",
            how="left",
            suffix="_period"
        )
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        .with_columns([
            pl.col("Perioden").str.slice(0, 4).alias("year")
        ])
        .group_by(["year", "Title_sector"])
        .agg([
            pl.col("Ziekteverzuimpercentage_1").mean().round(2).alias("avg_sick_leave_pct"),
            pl.count().alias("quarter_count"),
        ])
        .sort("year", descending=True)
    )

    return result


def data_quality_check():
    """Check for missing values by sector and period."""
    tables = load_80072_tables()

    result = (
        tables["fact"]
        .join(
            tables["dim_perioden"],
            left_on="Perioden",
            right_on="Key",
            how="left",
            suffix="_period"
        )
        .join(
            tables["dim_bedrijfskenmerken"],
            left_on="BedrijfskenmerkenSBI2008",
            right_on="Key",
            how="left",
            suffix="_sector"
        )
        .group_by(["Title_sector", "Perioden"])
        .agg([
            pl.count().alias("record_count"),
            pl.col("Ziekteverzuimpercentage_1").is_null().sum().alias("null_count"),
        ])
        .with_columns([
            (pl.col("null_count") / pl.col("record_count") * 100).round(1).alias("null_pct")
        ])
        .filter(pl.col("null_count") > 0)
        .sort("null_count", descending=True)
    )

    return result


# Example usage
if __name__ == "__main__":
    print("=== Example 1: Basic Join ===")
    df = basic_join()
    print(f"Shape: {df.shape}")
    print(df.head(10))

    print("\n=== Example 2: Time Series for All Economic Activities ===")
    df = time_series_by_sector()
    print(df.tail(10))

    print("\n=== Example 3: Sector Comparison (Latest Quarter) ===")
    df = sector_comparison_latest()
    print(df)

    print("\n=== Example 4: Aggregate by Category ===")
    df = aggregate_by_category()
    print(df)

    print("\n=== Example 5: Year-over-Year ===")
    df = year_over_year_comparison()
    print(df)
