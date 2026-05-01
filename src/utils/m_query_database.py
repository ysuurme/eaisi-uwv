"""
Purpose:    Reusable utility to query a SQLite database and return either a
            Polars or Pandas DataFrame.
"""
from pathlib import Path
from typing import Literal

import pandas as pd
import polars as pl
from sqlalchemy import create_engine

# --- Logging ---
from src.utils.m_log import f_log


def f_query_database(
    db_path: Path,
    query: str,
    return_type: Literal["polars", "pandas"],
) -> pl.DataFrame | pd.DataFrame:
    """
    Execute a SQL query against a SQLite database and return the result
    as either a Polars or Pandas DataFrame.

    Parameters
    ----------
    db_path : Path
        Absolute path to the SQLite database file (e.g. DIR_DB_GOLD, DIR_DB_SILVER).
    query : str
        SQL query to execute against the database.
    return_type : str, optional
        Format of the returned DataFrame. Either 'polars' (default) or 'pandas'.

    Returns
    -------
    pl.DataFrame | pd.DataFrame
        Query result in the requested format.

    Raises
    ------
    FileNotFoundError
        If the database file does not exist at the given path.
    ValueError
        If return_type is not 'polars' or 'pandas'.
    RuntimeError
        If the query fails for any reason.

    Examples
    --------
    from src.config import DIR_DB_GOLD, DIR_DB_SILVER
    from src.utils.m_query_database import f_query_database

    # Return a Polars DataFrame (default)
    df = f_query_database(
        db_path=DIR_DB_GOLD,
        query="SELECT * FROM prediction_baseline_total"
    )

    # Return a Pandas DataFrame
    df = f_query_database(
        db_path=DIR_DB_SILVER,
        query="SELECT * FROM 80072ned_silver WHERE year >= 2018",
        return_type="pandas"
    )
    """
    if return_type not in ("polars", "pandas"):
        raise ValueError(
            f"return_type must be 'polars' or 'pandas'. Got: '{return_type}'"
        )

    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Ensure the correct path is passed and the database has been initialised."
        )

    engine = create_engine(f"sqlite:///{db_path}")

    try:
        with engine.connect() as conn:
            if return_type == "polars":
                df = pl.read_database(query=query, connection=conn)
            else:
                df = pd.read_sql_query(query, conn)

        f_log(f"Query returned {len(df)} rows from {db_path.name} as {return_type} DataFrame", c_type="success")
        return df

    except Exception as e:
        raise RuntimeError(
            f"Query failed on {db_path.name}: {e}"
        ) from e

    finally:
        engine.dispose()