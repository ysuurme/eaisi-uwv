import logging
from pathlib import Path

import pandas as pd
import polars as pl
from sqlalchemy import create_engine, inspect, text

from src.config import DIR_DB_GOLD

logger = logging.getLogger(__name__)


def f_nb_results_to_gold_export(
    df: pl.DataFrame | pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
    dtype_map: dict | None = None,
) -> None:
    """
    Export a Polars or Pandas DataFrame to the gold SQLite database.

    Follows the same conventions as DatabaseGold in data_loader_gold.py:
    - Table names are stored as-is (caller is responsible for naming conventions).
    - The gold DB directory is created if it does not exist.
    - Existing tables are replaced by default; pass if_exists='append' to append.

    Parameters
    ----------
    df : pl.DataFrame | pd.DataFrame
        The DataFrame to export. Polars DataFrames are converted to Pandas
        automatically because SQLAlchemy's to_sql() requires Pandas.
    table_name : str
        Target table name in the gold database, e.g. 'predictions_baseline_total'.
    if_exists : str
        Behaviour when the table already exists: 'replace' (default), 'append',
        or 'fail'. Matches the pandas to_sql() if_exists semantics.
    dtype_map : dict | None
        Optional dictionary of {column_name: sqlalchemy_type} to enforce specific
        SQLite column types (e.g. {{'period_enddate': sqlalchemy.Date()}}).
        If None, types are inferred automatically.

    Raises
    ------
    ValueError
        If if_exists is not one of the accepted values.
    RuntimeError
        If the export fails for any reason.

    Examples
    --------
    # From the baseline notebook:
    export_to_gold(df_baseline, "baseline_predictions_gold")

    # From a model notebook, appending new model results:
    export_to_gold(df_results, "xgboost_predictions_gold", if_exists="replace")
    """
    if if_exists not in ("replace", "append", "fail"):
        raise ValueError(f"if_exists must be 'replace', 'append', or 'fail'. Got: '{if_exists}'")

    # --- Normalise input to Pandas ---
    if isinstance(df, pl.DataFrame):
        df_pd = df.to_pandas()
    elif isinstance(df, pd.DataFrame):
        df_pd = df.copy()
    else:
        raise TypeError(f"df must be a Polars or Pandas DataFrame, got {type(df)}")

    # --- Verify gold DB exists ---
    if not DIR_DB_GOLD.exists():
        raise FileNotFoundError(
            f"❌ Gold database not found at {DIR_DB_GOLD}. "
            "Run data_loader_gold.py first to initialise the database."
        )

    gold_path = DIR_DB_GOLD
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write to SQLite ---
    engine = create_engine(f"sqlite:///{gold_path}")

    try:
        df_pd.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            dtype=dtype_map,
        )
        n_rows = len(df_pd)
        logger.info(f"✅ Exported {n_rows} rows to table '{table_name}' in {gold_path}")
        print(f"✅ Exported {n_rows} rows → '{table_name}' ({gold_path.name})")

    except Exception as e:
        raise RuntimeError(f"Failed to export '{table_name}' to gold DB: {e}") from e

    finally:
        engine.dispose()


def f_list_gold_tables() -> list[str]:
    """
    Returns a list of all table names currently in the gold database.
    Useful for a quick sanity check after exporting.
    """
    if not DIR_DB_GOLD.exists():
        print(f"⚠️  Gold database does not exist yet at {DIR_DB_GOLD}")
        return []

    engine = create_engine(f"sqlite:///{DIR_DB_GOLD}")

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return tables
    finally:
        engine.dispose()