import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Baseline model
    ## Introduction
    This notebook contains the baseline model that we need to outperform. The baseline model is a weighted average (seasonal) timeseries forecast for the next 4 quarters.

    The data is limited to 2012 and onwards, so that we have a prediction for the last 10 years (2015-2025). This can be changed by adjusting the where clause in the SQL statement.

    In the bottom section of the notebook the baseline prediction is visualised for three groups of SBI codes:
    - Total aggregated absenteeism (sbi=T001081)
    - Absenteeism per SBI category (sbi lvl1, category A, B C etc.)
    - Absenteeism per company size (sbi=WP19098,WP19091, WP19078)

    ## Data preparation
    """)
    return


@app.cell
def _():
    import sys
    import os
    from pathlib import Path
    # Add project root to sys.path to allow imports from src
    sys.path.append(str(Path(os.getcwd()).parent.parent))

    import polars as pl
    import polars.selectors as cs
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sqlalchemy import create_engine
    from src.config import DIR_DB_SILVER, DIR_DATA_RAW
    from src.utils.m_nb_results_to_gold_export import f_nb_results_to_gold_export, f_list_gold_tables
    from src.utils.m_query_database import f_query_database
    from src.utils.m_sbi_classifier import f_split_by_sbi

    # Settings
    pl.Config(
        tbl_cols=-1,        # Show all columns
        tbl_rows=20,        # Show 20 rows
        tbl_width_chars=10000  # Prevent truncation due to width limit
    )

    # SQL Query to extract and format absenteeism data
    query = """ 
    SELECT 
        Perioden as timeperiod_text, 
        BedrijfskenmerkenSBI2008  as sbi_code,
        BedrijfskenmerkenSBI2008_Title as sbi_title,
        DATE(
            printf('%s-%s-01', 
                substr(Perioden, 1, 4), 
                CASE substr(Perioden, 7, 2)
                    WHEN '01' THEN '01'
                    WHEN '02' THEN '04'
                    WHEN '03' THEN '07'
                    WHEN '04' THEN '10'
                END
            ), 
            '+3 months', 
            '-1 day'
        ) AS period_enddate,
        CAST(substr(Perioden, 1, 4) as INTEGER) as "year",
        CAST(substr(Perioden, 8, 1) as INTEGER) as "quarter",
        CAST(Ziekteverzuimpercentage_1 AS REAL) as absenteeism_perc
    FROM "80072ned_silver"
    WHERE Perioden NOT LIKE '%JJ%' 
    AND substr(Perioden, 1, 4) >= '2012'
    order by sbi_code, Period_enddate asc
    """
    df_org = f_query_database(DIR_DB_SILVER, query, "polars")

    print(f"✅ Success! Loaded {len(df_org)} rows.")
    df_org.head()
    return (
        DIR_DATA_RAW,
        Path,
        cs,
        df_org,
        f_list_gold_tables,
        f_nb_results_to_gold_export,
        f_split_by_sbi,
        pd,
        pl,
        plt,
        sns,
    )


@app.cell
def _(cs, df_org):
    df_modified = df_org.with_columns(
        # Convert columns ending with 'date' to Date type
        cs.ends_with("date").str.to_date("%Y-%m-%d"),
    )
    df_modified.sort("period_enddate", descending=True).head()
    return (df_modified,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Year-on-Year Moving Average Prediction Model
    **Goal:** <br> For each quarter (Q1/Q2/Q3/Q4) and sbi_code, calculate a rolling 3-year moving average of Absenteeism_perc as a simple prediction.
    Example: The prediction for Q1-2019 = average of Q1-2016, Q1-2017, Q1-2018.

    **Step 1: Sort the Data** <br>
    Polars window functions respect row order, so we must sort by sbi_code, quarter, and year to ensure the rolling average looks back over the correct preceding years (e.g. 2016 → 2017 → 2018 for a Q1 prediction of 2019).
    """)
    return


@app.cell
def _(df_modified):
    df_sorted = df_modified.sort(["sbi_code", "quarter", "year"])
    df_sorted.head()
    return (df_sorted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Step 2: Calculate the Rolling 3-Year Moving Average per Quarter and sbi_code** <br>
    Let $\hat{y}_{t}$ be the predicted absenteeism rate for the current quarter, and $y_{t-n}$ be the actual quartly absenteeism rate from $n$ years ago, the formula is:$$\hat{y}_{t} = \frac{y_{t-1} + y_{t-2} + y_{t-3}}{3}$$(Where $t$ represents the same quarter in previous years).

    We use .over(["sbi_code", "quarter"]) to partition the data into groups, so the rolling average is calculated independently for each unique combination of sbi_code and Quarter (e.g. all Q1 rows for sbi_code "A").

    Explanation of the calculation:<br>
    - shift(1): This prevents "Look-ahead Bias." It ensures that the value for the current row is ignored, and the window starts from the previous year's data
    - window_size=3: averages the 3 most recent years for that quarter *starting from the previous year's data*
    - min_periods=1 — allows the average to be computed even when fewer than 3 years of history are available (e.g. the first year in the data)
    """)
    return


@app.cell
def _(df_sorted, pl):
    # 1. Apply the shift and rolling mean within the seasonal groups
    df_baseline = df_sorted.with_columns(
        pl.col("absenteeism_perc")
        .shift(1) # Moves previous years' data into the current year's row
        .rolling_mean(window_size=3, min_samples=3)
        .over(["sbi_code", "quarter"]) # Ensures we only average the same quarter across years
        .alias("baseline_prediction")
    )
    # 2. Calculate the Residual (Error) to evaluate the baseline
    df_baseline = df_baseline.with_columns(
        (pl.col("absenteeism_perc") - pl.col("baseline_prediction")).alias("residual_error")
        ).with_columns(pl.col("residual_error").abs().alias("abs_error"))
    # Preview the results ordered by period_enddate
    df_baseline.sort(["sbi_code", "period_enddate"]).head(10)

    # Uncomment to preview the results for a specific sbi_code
    # df_baseline.filter(pl.col("sbi_code") == "301000").sort("period_enddate").head(43)
    return (df_baseline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Split dataset into three seperate dataframes** <br>
    - df_total: containing the total absenteeism as calculated by the CBS
    - df_sbi_lvl1: containing the absenteeism grouped by the first SBI code level [A-U]
    - df_compsize: containing the absenteeism grouped by company size
    """)
    return


@app.cell
def _(DIR_DATA_RAW, Path, df_baseline, f_split_by_sbi, pl):
    # --- Old approach: manual filtering per SBI level ---
    # df_total = df_baseline.filter(pl.col("sbi_code") == "T001081")
    # df_sbi_lvl1 = df_baseline.filter(pl.col("sbi_title").str.to_uppercase().str.contains(r"^[A-U]\s"))
    # df_compsize = df_baseline.filter(pl.col("sbi_code") .is_in(["WP19098", "WP19091", "WP19078"]))

    # --- New approach: automated split using f_split_by_sbi ---
    # The function auto-detects CBS internal keys and splits into one DataFrame
    # per hierarchy level using the CBS dimension JSON as reference.
    # Note: f_split_by_sbi works with pandas, so we convert Polars -> Pandas -> Polars.
    dim_json_path = Path(DIR_DATA_RAW) / "80072ned" / "BedrijfskenmerkenSBI2008.json"

    sbi_splits = f_split_by_sbi(
        df=df_baseline.to_pandas(),
        sbi_column="sbi_code",
        dimension_json_path=dim_json_path,
    )

    # Convert each split back to Polars and assign to named variables
    df_total    = pl.from_pandas(sbi_splits["df_totaal"])
    df_sbi_lvl1 = pl.from_pandas(sbi_splits["df_section"])
    df_compsize = pl.from_pandas(sbi_splits["df_size"])

    # Show all available splits and their row counts
    print("Available splits:")
    for name, sub_df in sbi_splits.items():
        print(f"  {name}: {len(sub_df)} rows")

    df_total.head()
    return df_compsize, df_sbi_lvl1, df_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Store model output in Gold database
    Store the baseline model prediction output in gold database
    """)
    return


@app.cell
def _(
    df_compsize,
    df_sbi_lvl1,
    df_total,
    f_list_gold_tables,
    f_nb_results_to_gold_export,
):
    export_cols = [
        "sbi_code", "sbi_title", "year", "quarter", "period_enddate",
        "absenteeism_perc", "baseline_prediction", "residual_error", "abs_error",
    ]

    exports = {
        "prediction_baseline_total"  : df_total,
        "prediction_baseline_sbi"    : df_sbi_lvl1,
        "prediction_baseline_compsize": df_compsize,
    }

    for table_name, df in exports.items():
        f_nb_results_to_gold_export(
            df=df.select(export_cols),
            table_name=table_name,
        )

    # Sanity check — prints all tables in gold DB
    print("Tables in gold DB:", f_list_gold_tables())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation
    To calculate the Mean Absolute Error (MAE), we need to determine how far off our predictions are from the actual values on average, regardless of whether the prediction was too high or too low. By taking the Absolute Value, we ensure every "miss" is treated as a positive distance from the truth.
    """)
    return


@app.cell
def _(df_compsize, df_sbi_lvl1, df_total, pl):
    # 1. Calculate the Global MAE
    total_mae = df_total.select(pl.col("abs_error").mean()).item()

    print(f"Global Mean Absolute Error: {total_mae:.4f}")

    # 2. Calculate MAE per SBI_code (Industry)
    # This helps identify if the baseline works better for some industries than others
    mae_per_industry = (
        df_sbi_lvl1
        .group_by("sbi_code")
        .agg(
            pl.col("abs_error").mean().alias("mae")
        )
        .sort("mae")
    )
    # 2. Calculate MAE per company size
    mae_per_compsize = (
        df_compsize
        .group_by("sbi_code")
        .agg(
            pl.col("abs_error").mean().alias("mae")
        )
        .sort("mae")
    )
    # Convert MAE per industry to a dict for easy lookup
    mae_dict_sbi = dict(zip(mae_per_industry["sbi_code"].to_list(), mae_per_industry["mae"].to_list()))
    # Convert MAE per company size to a dict for easy lookup
    mae_dict_compsize = dict(zip(mae_per_compsize["sbi_code"].to_list(), mae_per_compsize["mae"].to_list()))
    print(mae_per_industry.head())
    print(mae_per_compsize.head())
    return mae_dict_compsize, mae_dict_sbi, total_mae


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisation
    **Step 1: Prepare the Data for Plotting** <br>
    Polars dataframes need to be converted to Pandas for Seaborn to work with them. We also sort by date to ensure the lines are drawn in chronological order.
    """)
    return


@app.cell
def _(df_compsize, df_sbi_lvl1, df_total, pd, pl):
    # Function to convert Polars dataframe to Pandas and sort chronologically for correct line drawing
    # because seaborn works with Pandas
    def to_plot_df(df: pl.DataFrame) -> pd.DataFrame:
        return (
            df
            .select(["timeperiod_text", "period_enddate", "sbi_code", "sbi_title", "baseline_prediction", "absenteeism_perc", "abs_error"])
            .drop_nulls()  # Remove the first year per group, which has no prediction due to the shift
            .to_pandas()
            .sort_values(["sbi_code", "period_enddate"])
            .reset_index(drop=True)
        )

    df_plot_total    = to_plot_df(df_total)
    df_plot_sbi      = to_plot_df(df_sbi_lvl1)
    df_plot_compsize = to_plot_df(df_compsize)

    df_plot_total.head(10)
    return df_plot_compsize, df_plot_sbi, df_plot_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Step 2: Plot the Time Series Total**<br>
    """)
    return


@app.cell
def _(df_plot_total, pd, plt, sns, total_mae):
    # --- 1. Define the COVID Period boundaries ---
    covid_start = pd.Timestamp('2020-03-31') # Q1 2020
    covid_end = pd.Timestamp('2022-06-30')   # Q2 2022

    # --- 2. Prepare the figure ---
    (_fig, _ax) = plt.subplots(figsize=(12, 6))

    # Plot Predicted Absenteeism
    sns.lineplot(data=df_plot_total, x='period_enddate', y='baseline_prediction', 
                 label='Predicted Absenteeism', marker='o', errorbar=None, 
                 ax=_ax, color='tab:blue', linewidth=2.5, markersize=8)

    # Plot Actual Absenteeism
    sns.lineplot(data=df_plot_total, x='period_enddate', y='absenteeism_perc', 
                 label='Actual Absenteeism', marker=None, errorbar=None, 
                 ax=_ax, color='tab:orange', linewidth=2, linestyle='--', alpha=0.7)

    # MAE Band
    _df_band = df_plot_total.groupby('period_enddate')['baseline_prediction'].mean().reset_index()
    _ax.fill_between(_df_band['period_enddate'], 
                     _df_band['baseline_prediction'] - total_mae, 
                     _df_band['baseline_prediction'] + total_mae, 
                     color='tab:blue', alpha=0.15, label=f'Error bandwidth (±{total_mae:.2f}%)', edgecolor='none')

    # --- 3. Add the Vertical Band (COVID Period) ---
    _ax.axvspan(covid_start, covid_end, color='grey', alpha=0.1)
    _ax.text(x=covid_start + (covid_end - covid_start) / 2, 
             y=0.95, 
             s='COVID Period', 
             color='dimgrey', 
             fontweight='bold',
             ha='center', 
             va='top', 
             transform=_ax.get_xaxis_transform())

    # --- 3.5. Add Labels for the LAST 12 Predicted markers ---
    # We sort to ensure we are actually getting the most recent dates
    _df_last_12 = df_plot_total.sort_values('period_enddate').iloc[-12:]

    for _, _row in _df_last_12.iterrows():
        _ax.annotate(
            text=f"{_row['baseline_prediction']:.1f}%", 
            xy=(_row['period_enddate'], _row['baseline_prediction']),
            xytext=(0, 12),           # 10 points vertical offset
            textcoords='offset points',
            ha='center', 
            va='bottom',
            fontsize=10,
            fontweight='bold',
            color='tab:blue'
        )

    # --- 4. Styling & Labels ---
    _ax.set_title('Overall Actual vs. Predicted Average Absenteeism Rate', fontsize=14, fontweight='bold')
    _ax.set_xlabel('Quarter End Date')
    _ax.set_ylabel('Absenteeism (%)')

    _unique_dates = sorted(df_plot_total['period_enddate'].unique())
    _tick_labels = [f'Q{(d.month - 1) // 3 + 1} {d.year}' for d in _unique_dates]
    _ax.set_xticks(_unique_dates)
    _ax.set_xticklabels(_tick_labels, rotation=45, ha='right')

    _ax.set_ylim(df_plot_total['absenteeism_perc'].min() * 0.8, df_plot_total['baseline_prediction'].max() * 1.3)
    _ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.show()
    return covid_end, covid_start


@app.cell
def _(covid_end, covid_start, df_plot_total, plt, sns, total_mae):
    #Viz code for total MAE per quarter
    (_fig, _ax) = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_plot_total, x='period_enddate', y='abs_error', marker='o', color='tab:blue', linewidth=2.5, markersize=8, ax=_ax)
    for (_, _row) in df_plot_total.iterrows():
        _ax.annotate(text=f"{_row['abs_error']:.2f}%", xy=(_row['period_enddate'], _row['abs_error']), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color='tab:blue')
    _ax.axhline(total_mae, color='tab:red', linestyle='--', linewidth=1.5, label=f'Total MAE ({total_mae:.2f}%)')
    _unique_dates = sorted(df_plot_total['period_enddate'].unique())
    _tick_labels = [f'Q{(d.month - 1) // 3 + 1} {d.year}' for d in _unique_dates]
    _ax.axvspan(covid_start, covid_end, color='grey', alpha=0.1)
    _ax.text(x=covid_start + (covid_end - covid_start) / 2, 
             y=0.95, 
             s='COVID Period', 
             color='dimgrey', 
             fontweight='bold',
             ha='center', 
             va='top', 
             transform=_ax.get_xaxis_transform())
    _ax.set_xticks(_unique_dates)
    _ax.set_xticklabels(_tick_labels, rotation=45, ha='right')
    _ax.set_title('Mean Absolute Error - Total absenteeism per Quarter', fontsize=14, fontweight='bold')
    _ax.set_xlabel('Quarter End Date')
    _ax.set_ylabel('Mean Absolute Error (%)')
    _ax.legend(loc='upper left')
    # Annotate each point
    plt.tight_layout()
    # Add global MAE as a reference line
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Step 3: Plot the Time Series per SBI code**<br>
    First chart shows the MAE per SBI Code (very crowded graph, need to fix that)
    Below this are charts per SBI code that show prediction vs. actual.
    """)
    return


@app.cell
def _(df_plot_sbi, plt, sns, total_mae):
    #Viz code for MAE per SBI code over quarter
    (_fig, _ax) = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=df_plot_sbi, x='period_enddate', y='abs_error', hue='sbi_code', marker='o', linewidth=2, markersize=6, errorbar=None, ax=_ax)
    _ax.axhline(total_mae, color='black', linestyle='--', linewidth=1.5, label=f'Total MAE ({total_mae:.2f}%)')
    _unique_dates = sorted(df_plot_sbi['period_enddate'].unique())
    _tick_labels = [f'Q{(d.month - 1) // 3 + 1} {d.year}' for d in _unique_dates]
    _ax.set_xticks(_unique_dates)
    _ax.set_xticklabels(_tick_labels, rotation=45, ha='right')
    _ax.set_title('MAE per SBI Code over Quarter', fontsize=14, fontweight='bold')
    _ax.set_xlabel('Quarter End Date')
    _ax.set_ylabel('Mean Absolute Error (%)')
    _ax.legend(loc='upper left', title='SBI Code')
    plt.tight_layout()
    # Add global MAE as a reference line
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visuals per SBI code
    """)
    return


@app.cell
def _(covid_end, covid_start, df_plot_sbi, mae_dict_sbi, plt, sns):
    #Viz code for Actual vs Predicted per SBI code with MAE bandwidth
    _sbi_codes = sorted(df_plot_sbi['sbi_code'].unique())
    for _sbi_code in _sbi_codes:
        _df_sbi = df_plot_sbi[df_plot_sbi['sbi_code'] == _sbi_code].copy().reset_index(drop=True)
        _mae = mae_dict_sbi.get(_sbi_code, 0)
        _sbi_title = _df_sbi['sbi_title'].iloc[0]
    
        (_fig, _ax) = plt.subplots(figsize=(12, 6))
    
        # 1. Plot Predicted Absenteeism
        sns.lineplot(data=_df_sbi, x='period_enddate', y='baseline_prediction', 
                     label='Predicted Absenteeism', marker='o', errorbar=None, 
                     ax=_ax, color='tab:blue', linewidth=2.5, markersize=8)
    
        # 2. Plot Actual Absenteeism
        sns.lineplot(data=_df_sbi, x='period_enddate', y='absenteeism_perc', 
                     label='Actual Absenteeism', marker=None, errorbar=None, 
                     ax=_ax, color='tab:orange', linewidth=2, linestyle='--', alpha=0.7)

        # 3. Add the COVID Vertical Band
        _ax.axvspan(covid_start, covid_end, color='grey', alpha=0.10)
    
        # Add Label centered at the top of the band
        _ax.text(x=covid_start + (covid_end - covid_start) / 2, 
                 y=0.95, 
                 s='COVID Period', 
                 color='dimgrey', 
                 fontweight='bold',
                 ha='center', 
                 va='top', 
                 transform=_ax.get_xaxis_transform())

        # 4. Annotate markers for Predicted line
        for (_, _row) in _df_sbi.iterrows():
            _ax.annotate(text=f"{_row['baseline_prediction']:.1f}%", 
                         xy=(_row['period_enddate'], _row['baseline_prediction']), 
                         xytext=(0, 12), textcoords='offset points', 
                         ha='center', va='bottom', fontsize=9, fontweight='bold', color='tab:blue')

        # 5. MAE Bandwidth
        _df_band = _df_sbi.sort_values('period_enddate')
        _ax.fill_between(_df_band['period_enddate'], 
                         _df_band['baseline_prediction'] - _mae, 
                         _df_band['baseline_prediction'] + _mae, 
                         color='tab:blue', alpha=0.15, label=f'Error bandwidth (±{_mae:.2f}%)', edgecolor='none')

        # 6. Formatting
        _ax.set_title(f'{_sbi_title} — Actual vs. Predicted Absenteeism Rate', fontsize=14, fontweight='bold')
        _ax.set_xlabel('Quarter End Date')
        _ax.set_ylabel('Absenteeism (%)')

        _unique_dates = sorted(_df_sbi['period_enddate'].unique())
        _tick_labels = [f'Q{(d.month - 1) // 3 + 1} {d.year}' for d in _unique_dates]
        _ax.set_xticks(_unique_dates)
        _ax.set_xticklabels(_tick_labels, rotation=45, ha='right')

        # Buffering the y-limit slightly more to account for the top label
        _ax.set_ylim(_df_sbi['absenteeism_perc'].min() * 0.8, _df_sbi['baseline_prediction'].max() * 1.3)
        _ax.legend(loc='upper left', frameon=True)

        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Visuals per Company size
    """)
    return


@app.cell
def _(covid_end, covid_start, df_plot_compsize, mae_dict_compsize, plt, sns):
    #Viz code for Actual vs Predicted per SBI code with MAE bandwidth
    _sbi_codes = sorted(df_plot_compsize['sbi_code'].unique())
    for _sbi_code in _sbi_codes:
        _df_sbi = df_plot_compsize[df_plot_compsize['sbi_code'] == _sbi_code].copy().reset_index(drop=True)
        _mae = mae_dict_compsize.get(_sbi_code, 0)
        _sbi_title = _df_sbi['sbi_title'].iloc[0]
    
        (_fig, _ax) = plt.subplots(figsize=(12, 6))
    
        # 1. Plot Predicted & Actual Lines
        sns.lineplot(data=_df_sbi, x='period_enddate', y='baseline_prediction', label='Predicted Absenteeism', marker='o', errorbar=None, ax=_ax, color='tab:blue', linewidth=2.5, markersize=8)
        sns.lineplot(data=_df_sbi, x='period_enddate', y='absenteeism_perc', label='Actual Absenteeism', marker=None, errorbar=None, ax=_ax, color='tab:orange', linewidth=2, linestyle='--', alpha=0.7)
    
        # --- ADDED: COVID Vertical Bandwidth ---
        _ax.axvspan(covid_start, covid_end, color='grey', alpha=0.15, label='COVID Period')
    
        # Add Label centered at the top end of the band
        # transform=_ax.get_xaxis_transform() pins the Y-coordinate to 0.95 (95% of the way up the axis)
        _ax.text(x=covid_start + (covid_end - covid_start) / 2, 
                 y=0.95, 
                 s='COVID Period', 
                 color='dimgrey', 
                 fontweight='bold',
                 ha='center', 
                 va='top', 
                 transform=_ax.get_xaxis_transform())
        # ----------------------------------------

        # 2. Annotate markers for Predicted line
        for (_, _row) in _df_sbi.iterrows():
            _ax.annotate(text=f"{_row['baseline_prediction']:.1f}%", xy=(_row['period_enddate'], _row['baseline_prediction']), xytext=(0, 12), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold', color='tab:blue')
    
        # 3. MAE Bandwidth
        _df_band = _df_sbi.sort_values('period_enddate')
        _ax.fill_between(_df_band['period_enddate'], _df_band['baseline_prediction'] - _mae, _df_band['baseline_prediction'] + _mae, color='tab:blue', alpha=0.15, label=f'MAE Band (±{_mae:.2f}%)', edgecolor='none')
    
        # 4. Formatting
        _ax.set_title(f'SBI Code: {_sbi_title} — Actual vs. Predicted Absenteeism Rate', fontsize=14, fontweight='bold')
        _ax.set_xlabel('Quarter End Date')
        _ax.set_ylabel('Absenteeism (%)')
    
        _unique_dates = sorted(_df_sbi['period_enddate'].unique())
        _tick_labels = [f'Q{(d.month - 1) // 3 + 1} {d.year}' for d in _unique_dates]
        _ax.set_xticks(_unique_dates)
        _ax.set_xticklabels(_tick_labels, rotation=45, ha='right')
    
        # Adjusted y-limit to 1.3 to ensure the COVID label doesn't overlap with peaks
        _ax.set_ylim(_df_sbi['absenteeism_perc'].min() * 0.8, _df_sbi['baseline_prediction'].max() * 1.3)
        _ax.legend(loc='upper left', frameon=True)
    
        plt.tight_layout()
        plt.show()
    return


if __name__ == "__main__":
    app.run()
