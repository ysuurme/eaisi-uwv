# EAISI UWV

## Table of Contents

- [🚀 Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Project Contributions](#project-contributions)
- [GitHub Flow](#github-flow)
- [Project Management](#project-management)
- [Data Management](#data-management)

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/EAISI/eaisi-uwv.git
cd eaisi-uwv
# Sync dependencies using uv (https://astral.sh/uv)
uv sync
# Install project in editable mode to resolve imports (Run this once per environment)
uv pip install -e .
```

### 2. Execute Orchestration Pipeline
The `main.py` entrypoint serves as the holistic orchestrator for both Data Engineering and Machine Learning.

**Usage:**
```
uv run main.py [gold_table] [model_key] [sbi_filter_col] [group1,group2,...]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `gold_table` | `master_data_ml_preprocessed` | Gold feature store table name |
| `model_key` | `baseline` | Key from the ModelConfiguration catalog |
| `sbi_filter_col` | *(omit)* | OHE column to isolate one sector, e.g. `BedrijfskenmerkenSBI2008_301000`. Omit for national total (T001081). Use `-` as a placeholder to skip when specifying feature groups. |
| `feature_groups` | *(omit)* | Comma-separated catalog group names, e.g. `compensation,labor_volume` |

**Examples:**

```bash
# National total (T001081) — baseline model, all-industry 4Q forecast
uv run main.py master_data_ml_preprocessed baseline

# Single sector — baseline, sector 301000
uv run main.py master_data_ml_preprocessed baseline BedrijfskenmerkenSBI2008_301000

# All sectors at once — one MLflow run per sector, same experiment
uv run main.py master_data_ml_preprocessed baseline --all-sectors

# Ridge (feature-ML), skip the SBI arg with '-', use specific catalog feature groups
uv run main.py master_data_ml_preprocessed ridge - labor_structure,wages
```

**Lifecycle verbs** — the whole flow runs from `main.py`:

```bash
uv run main.py --refresh-data        # raw → bronze → silver → gold
uv run main.py --select-features     # gold → statistical funnel → feature_catalog.json
uv run main.py --full-sweep          # every model family × all sectors → eval DB (the "clean run")
uv run main.py --compare             # cross-method scorecard + decision matrix → reports/comparison/
uv run main.py --forecast            # forward 4Q from every @prod champion → model_forecasts + figures
uv run main.py --report              # leaderboard, winners quadrant, figures/CSVs, narrative summary
```

**Available model keys:** `baseline`, `autoets`, `stl_ets`, `chronos_bolt`, `ridge`, `random_forest`, `ridge_deseason` (curated comparison set — see the Estimator Catalog below)

### 3. Track Results (MLflow)
The pipeline is "Zero-Artifact"; all results are stored in `data/4_eval/eval_data.db`. Launch the UI to view metrics, tuning grids, and model signatures.

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/4_eval/eval_data.db
# or the project script (see [project.scripts] in pyproject.toml):
uv run mlflow-ui
```

🌐 **Open**: [http://127.0.0.1:5000](http://127.0.0.1:5000)

All runs land in the `master_SickLeave_4Q` experiment, tagged with `sector` and `forecast_horizon=4Q` for easy cross-sector comparison. The business headline is **two numbers** — *is it good?* (MASE) and *how far off?* (MAE):

- **MASE** (Mean Absolute Scaled Error — outer-fold MAE scaled by the in-sample seasonal-naive m=4 MAE) — **THE comparison metric** and quality gate: scale-free and comparable across sectors with different baseline difficulty, lower is better, and **MASE < 1 beats the seasonal naive**.
- **MAE** (in percentage points) — the primary **stakeholder** magnitude: how far off, on average, in the same units as the target (sick-leave %).
- **MAPE** (relative `|y−ŷ|/|y|`) — retained as a **diagnostic** (eval DB + `reports/sector_quality.csv`), not part of the headline; MASE already gives the scale-free cross-sector view.

Each sector has one registered model whose `@prod` alias is its current champion (the lowest-MASE model seen so far); R²/RMSE are recorded alongside as secondary diagnostics.

### 4. Run Tests
```bash
uv run pytest
```

---
**Note**: To auto-launch the MLflow UI during training, set `START_MLFLOW_UI = True` in `config.py`.

This project is built using CPython 3.10.11 and dependencies are managed with UV. We use GitHub as our central hub for code management and project coordination. We use MLflow as a tool for managing the machine learning lifecycle.

References:
- UV CPython installation and dependency management: https://docs.astral.sh/uv/
- Github docs: https://docs.github.com/
- Project Structure: https://github.com/ysuurme/eaisi-uwv
- Project Contributions: https://docs.github.com/en/get-started/using-github/github-flow
- MLflow: https://mlflow.org/docs/latest/ml/getting-started/quickstart/

## Project Structure

- **Main Branch**: `main` - the stable, production-ready code
- **Feature Branches**: `feature/*` - for new features and enhancements
- **Bug Fix Branches**: `bugfix/*` - for bug fixes
- **Documentation Branches**: `docs/*` - for documentation updates

## Directory Layout

```text
eaisi-uwv/
├── data/                       # Medallion + eval SQLite DBs (0_raw … 4_eval, feature_selection); gitignored
├── reports/                    # Exported figures + CSVs (--report); reports/comparison/ (--compare)
├── notebooks/                  # Non-production experimentation (reference, no-copy; _legacy/ = retired)
│   ├── data_exploration/       # EDA and schema analysis
│   └── ml_experimentation/     # Model experiments + cv_output parquets
├── src/                        # Production source code
│   ├── data_engineering/       # Medallion pipeline: data_loader_{raw,bronze,silver,gold}.py
│   ├── ml_engineering/         # ML lifecycle — numbered 7-step pipeline + orchestrator + catalog
│   │   ├── ml_orchestrator.py  # Hub: run_pipeline / run_sector_sweep / run_full_sweep /
│   │   │                       #      run_feature_selection / run_forecast / run_comparison / run_report
│   │   ├── ml_1_data_extraction.py  …  ml_7_model_inference.py
│   │   └── model_configs.py    # Estimator catalog, ORM tables, SectorQuarterRollingMean baseline
│   ├── utils/                  # m_* helpers — m_evaluation, m_pipeline_loader, m_model_viz,
│   │                           #   m_sector_quality, m_log, m_query_database, m_sbi_classifier, …
│   └── config.py               # Project-wide config + CBS_TABLE_REGISTRY
├── main.py                     # CLI entry point (orchestration)
├── CONTEXT.md                  # ← architecture & navigation index (start here)
├── pyproject.toml              # Dependencies (uv)
└── README.md
```

→ See **[CONTEXT.md](CONTEXT.md)** for the authoritative Module Map, bounded contexts, and the Issue-Type → Files index.

## Project Contributions

We collaborate using GitHub as our central hub for code management and project coordination. All team members work together through:

- **GitHub Repository**: Our single source of truth for all code
- **Pull Requests**: Code review and integration mechanism
- **GitHub Projects**: Tracking and planning of issues (project tasks) in Kanban style
- **Issues**: Issue (project tasks) creation, description, assignment and planning
- **Wiki**: Code documentation

## GitHub Flow

Our development process follows the GitHub Flow model for a smooth and collaborative workflow:
- https://docs.github.com/en/get-started/using-github/github-flow

### Github Flow Steps

1. **Create a Branch**
   - Create a feature branch from `main` for each task or bug fix
   - Use descriptive branch names: `feature/user-authentication`, `bugfix/login-error`, `docs/readme-update`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Commit your changes with clear, descriptive commit messages
   - Keep commits atomic and focused on a single concern
   ```bash
   git commit -m "Add user authentication module"
   ```

3. **Push and Create a Pull Request**
   - Push your branch to the remote repository
   - Open a Pull Request (PR) on GitHub with a detailed description
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Code Review**
   - Team members review your PR for code quality, logic, and style
   - Address feedback and make requested changes
   - Discussions happen within the PR for transparency

5. **Merge to Main**
   - Once approved, merge your PR into `main`
   - Delete the feature branch after merging
   - Monitor for any issues post-merge

### Pull Request Best Practices

- **Title**: Be clear and concise (e.g., "Add login validation")
- **Description**: Explain what you changed and why
- **Linked Issues**: Reference related issues with `#issue-number`
- **Tests**: Include relevant tests for your changes
- **Screenshots**: Add visual proof for UI changes if applicable

## Project Management

We use a **GitHub Project** @project-eaisi-uwv to organize our work, track progress, and plan issues.

### How We Use GitHub Projects

- **Issue Organization**: Features, bugs, and documentation are tracked as issues and organized in @project-eaisi-uwv
- **Status Tracking**: Issues move through columns: `Todo` → `In Progress` → `In Review` → `Done`
- **Priority & Assignment**: Issues are prioritized and assigned to team members

### Getting Started with Issues

1. Check the current **GitHub Project** board for available issues
2. Find a task in the `Todo` that interests you
3. Assign yourself to the issue
4. Move it to `In Progress` when you have created a branch and start working
5. Create a Pull Request linked to the issue to track updates
6. Move to `In Review` when the Pull Request is ready
7. Move to `Done` after the Pull Request branch is merged to main

## Data Engineering
This documentation outlines the architectural strategy for our data pipeline, utilizing a Mini-Medallion Architecture implemented with Python 3, SQLAlchemy Core & ORM, and SQLite3.
The goal is to provide a lightweight, portable, and reproducible ELT (Extract, Load, Transform) framework for our machine learning project.

**Raw: The "Unstructured Zone"**
- Technology: Local project directory (/data/0_Raw/).
- Strategy: Original files (.json, .csv, etc.) exactly as received from APIs.
Excluded from Git (.gitignore) to prevent unnecessary data storage in the repository. Acts as the "Source of Truth" for debugging and re-processing.

**Bronze: The "Landing Zone"**; our first structured representation of the raw data.
- Technology: SQLite3 database inserts via SQLAlchemy Core (optimized for high-speed bulk inserts).
- Strategy: We implement a Star Schema at this level in the form of Fact (the main data) and each Dimension (the lookup tables).
   - Fact Tables: Store the main numerical data (e.g., TypedDataSet).
   - Dimension Tables: Store lookup data (e.g., Periods, Gender, PersonalCharacteristics).
   - Upsert Logic: "Upsert" (Update or Insert) so you only process new files in the 0_Raw folder rather than re-processing everything every time.

**Silver: The "Clean Zone"**; our second structured representation of data that focuses on data quality and integration.
- Technology: SQLite3 database inserts via SQLAlchemy ORM (better for complex logic).
- Strategy: Transform separate Fact and Dimension tables into a single, "meaningful" table by:
   - Flattening: Extracting nested JSON structures into flat columns
   - DataType Casting: Converting strings to proper DATETIME or NUMERIC types
   - Standardization: Trimming whitespace (e.g., "14   " $\rightarrow$ "14") and handling NULL values
   - Enrichment: Performing JOIN operations between Facts and Dimensions to replace cryptic codes (e.g., GM9001) with human-readable titles (Bonaire).

**Gold: The "Business/ML Zone"**; our third structured representation that provides clean features for analysis and ML training.
- Technology: SQLite3 database inserts via SQLAlchemy ORM.
- Strategy: Feature engineering, strict temporal standardization, and explicit Star Schema ML configuration:
   - **Target Data (Fact Tables)**: We conceptually do not want foundational dimensions (like Branches) pivoted under any circumstances for Target datasets (e.g., `80072ned`). Your model predicts sick leave *per branch*. This means your ML algorithm needs discrete row objects natively modeled around the specific `(Quarter, Branch)` composite key so it can scale iterations appropriately!
   - **Feature Data (Dimension Tables)**: Purely observational feature datasets (e.g., `85916NED`, `85920NED`) are dynamically flattened (pivoted) across their demographic properties to enforce exactly 1 row per Quarter natively. This ensures they can `Left Join` seamlessly onto the Fact table without triggering Cartesian data explosions.
   - **Structural Joins**: Identifying keys (like SBI Branches) within feature tables are mathematically preserved as vertical indices alongside Time, guaranteeing flawless index merging against the specific target rows.
   - **Data Quality Gates**: Zero-Null policies are enforced automatically via interpolation routines, ensuring the Data Store evaluates exactly to machine-learning constraints natively prior to execution.
   - **SBI Sector Encoding**: Each of the 39 SBI sectors (plus the national total `T001081`) is encoded as a binary OHE column `BedrijfskenmerkenSBI2008_XXXXXX`. The column `BedrijfskenmerkenSBI2008_T001081 = 1` identifies the pre-computed national total row — it is NOT derived by averaging sector rows.

### Imputation Methodology
We mathematically enforce a strict Zero-Null policy using a Grouped Time-Series Strategy (`src/utils/m_imputation.py`). 
- **Target Variables**: Sorted chronologically by Branch and Date. Short temporal gaps are bridged via Forward Fill (`ffill`) to preserve sector-specific reality without artificial spikes. Structural gaps fallback to Median Imputation for safe central tendency.
- **Feature Variables**: Binary One-Hot Encoded flags are filled with `0` (absent). Continuous metrics fallback to column Medians. Missing-indicator flags (`_is_missing`) are automatically generated to allow models to learn from the absence of data itself.

## Machine Learning Engineering
This project follows a modular **MLOps Level 0** architecture structured into sequential steps. The business objective is to **forecast sick leave (absenteeism) 4 quarters ahead, per SBI sector**.

### Forecast Paradigm
All models operate as **4-quarter-ahead time-series forecasters**:
- The forecast horizon is **4 quarters** (1 year): every origin forecasts 4 quarters ahead.
- The held-out **evaluation window is the last 20 quarters** (`n_test_points=20`), evaluated by **walk-forward over 5 rolling origins** (each origin refits on the expanding history and forecasts the next 4 quarters). These 5 origins are split into inner (variant-selection) and outer (honest) folds; headline metrics use the **outer** folds only.
- The baseline and all competing models are evaluated identically on this held-out window, with production-honest future X (no covariate leakage).
- Hyperparameter tuning runs on the **training set only** via `ExpandingWindowSplitter(fh=[1,2,3,4], step_length=4, initial_window=40)` — one full year per CV fold, evaluating all 4 ahead-steps simultaneously.

### Operational Modes
Every pipeline run operates on exactly one quarterly time series (1 row per quarter):

| Mode | How | Series used |
|------|-----|-------------|
| **All-industry** (default) | Filters `BedrijfskenmerkenSBI2008_T001081 == 1` | CBS national total |
| **Sector-specific** | Filters `BedrijfskenmerkenSBI2008_<code> == 1` | One SBI sector |
| **Sector sweep** | `--all-sectors` flag | All 39+ OHE columns, one run each |

### Estimator Catalog (`model_configs.py`)

A curated 7-model comparison set spanning the univariate-vs-multivariate question (`chronos_bolt` added as a foundation-model contender). Feature groups resolve from `data/feature_selection/feature_catalog.json` (the single source of truth; `all_survivors` = the selected features).

| Key | Estimator | Uses selected features? | Tunable |
|-----|-----------|---|---------|
| `baseline` | `SectorQuarterRollingMean(n_years=3)` | — (reference, run per sector) | No |
| `autoets` | `AutoETS(sp=4)` | ❌ univariate (ignores X) | ETS error/trend/seasonal |
| `stl_ets` | `QuarterlyPeriodForecaster(STLForecaster + ETS)` | ❌ univariate (ignores X) | STL `seasonal`/`robust`, trend damping |
| `chronos_bolt` | `ChronosForecaster(amazon/chronos-bolt-base)` — zero-shot foundation model | ❌ univariate (ignores X) | none (zero-shot) |
| `ridge` | `make_reduction(Pipeline[Scaler→Ridge], window_length=4)` on `all_survivors` | ✅ multivariate linear | `window_length`, `alpha` |
| `random_forest` | `make_reduction(RandomForestRegressor, window_length=12)` on `all_survivors` | ✅ multivariate non-linear | `window_length`, `n_estimators`, `max_depth`, `min_samples_leaf` |
| `ridge_deseason` | `QuarterlyPeriodForecaster(Deseasonalizer(sp=4) → Ridge reducer)` on `all_survivors` | ✅ multivariate (deseasonalized) | `window_length`, `alpha` |

`baseline`/`autoets`/`stl_ets`/`chronos_bolt` are univariate — they forecast the sick-leave rate from its own past only, so feature selection does not affect them. `chronos_bolt` is a zero-shot Amazon Chronos-Bolt foundation model (pretrained T5; no training on our data — `fit` stores the context, `predict` returns the median quantile; requires `torch` + `chronos`, runs on CPU). `ridge`/`random_forest`/`ridge_deseason` are the multivariate ML models that leverage the selected CBS drivers, testing whether the exogenous features add value beyond the target's history. Linear-family estimators (`ridge`, `ridge_deseason`) use a `StandardScaler` pipeline; `random_forest` is scale-invariant.

**`SectorQuarterRollingMean`**: Domain-specific baseline. For each quarter Q, predicts the mean of Q from the previous 3 years using `shift(1).rolling_mean(window_size=3, min_samples=3).over([quarter])`. Requires 3 full prior-year observations before producing a prediction (matching the CBS notebook approach). Because `shift(1)` within the same-quarter group equals a 1-year shift, this is inherently a 4-quarter-ahead forecast.

### Pipeline Steps

**0. Orchestration (`ml_orchestrator.py`)**
- Central hub coordinating all steps. Manages DB session and Unit of Work.
- `run_pipeline()`: single sector/all-industry run.
- `run_sector_sweep()`: discovers all OHE SBI columns at runtime, runs `run_pipeline` once per sector.

**1. Data Extraction (`ml_1_data_extraction.py`)**
- Filters gold table to the target series (T001081 or specified sector OHE column).
- Drops all 39 OHE SBI columns after filtering.
- Supports feature group selection (`FEATURE_CATALOG`) or full discovery mode.
- Output: exactly **1 row per unique quarter date**.

**2. Data Validation (`ml_2_data_validation.py`)**
- Enforces schema, zero-nulls, float64 dtypes, and `period_enddate` parseability.

**3. Data Preparation (`ml_3_data_preparation.py`)**
- Temporal train/test split: the last `n_test=20` quarters (the walk-forward evaluation window) are held out; everything before the cutoff is the training set.
- Sets a `DatetimeIndex` for sktime compatibility.
- Outputs `X_train, X_test, y_train, y_test` + lineage metadata.

**4. Model Training (`ml_4_model_training.py`)**
- **Baseline path** (`SectorQuarterRollingMean`): standard sklearn `.fit(X, y)`, logged via `mlflow.sklearn.log_model`.
- **sktime path** (all others): `.fit(y=y_train, X=X_numeric)`, tuned via `ForecastingGridSearchCV` + `ExpandingWindowSplitter`, wrapped in a pyfunc for registry compatibility.
- Logs full **reproducibility lineage**: `experiment_key`, `model_name`, `model_type` (the underlying algorithm, unwrapped from the sktime reducer), `feature_groups`, `feature_catalog`, `best_params`, `param_grid`, a `feature_set_hash` tag, and a `features.json` artifact — so any run is reproducible and two runs of the same estimator are distinguishable from MLflow metadata alone.

**5. Model Evaluation (`ml_5_model_evaluation.py`)**
- Walk-forward (rolling-origin) evaluation with nested inner/outer folds; headline metrics are computed on the honest **outer** folds.
- Computes **MASE** (THE comparison metric — outer-fold MAE scaled by the in-sample seasonal-naive m=4 MAE) and **MAE** (percentage points — the stakeholder magnitude) as the headline pair, plus **MAPE**, R², RMSE as diagnostics, on the 4-quarter-ahead forecasts and logs them to MLflow (MASE as `mean_absolute_scaled_error`). Per-row predictions are persisted for cross-model comparison.

**6. Model Validation & Registry (`ml_6_model_validation.py`)**
- **MASE champion/challenger gate**: one registered model per sector (sector-keyed name). A challenger is promoted to `@prod` only if its MASE is finite and strictly lower than the incumbent champion's MASE — or seeded unconditionally when no champion exists yet. An optional `max_mase` ceiling (default disabled) additionally requires `MASE < max_mase` (e.g. 1.0 = must beat the seasonal naive). Losing runs stay diagnosable (`passed_gate=false`) but are not registered, so the registry only grows on genuine improvement.
- R² is recorded alongside (optional `r2_floor`, disabled by default). The promoted version **self-describes** via tags `mase` / `mae` / `mape` / `r2` / `model_family` / `model_type` / `feature_groups`, so the registry alone tells you each champion's comparison score, stakeholder accuracy (pp), algorithm, and the config features it used.

**7. Model Inference / Forward Forecast (`ml_7_model_inference.py`)**
- Resolves each sector's `@prod` champion, rebuilds it from its MLflow lineage, **refits on the full observed history**, and forecasts the next 4 quarters with production-honest future X. `main.py --forecast` persists these to `model_forecasts` and renders one overlay figure per sector.

### Cross-method comparison (`--compare`) and reporting (`--report`)
- **`--full-sweep`** runs every catalog family across all sectors so they all compete for each sector's `@prod` slot in one command.
- **`--compare`** (`m_pipeline_loader.load_families_from_eval_db` → `m_evaluation.compare_all_models`) aligns every family on the shared (sector, target_date, horizon) keys and writes a scorecard + decision matrix to `reports/comparison/`: point/regime metrics, skill vs the baseline, and Diebold-Mariano + Friedman/Nemenyi significance tests.
- **`--report`** regenerates the figure bundle — leaderboard, **winners quadrant** (each sector's champion by paradigm) and win counts, predicted-vs-actual, experiment matrix, horizon decay, forecast overlays — plus `reports/sector_quality.csv` and a narrative summary under `.claude/`.

### Sector Performance Read-Model (`m_sector_quality.py`)
MLflow is the single source of truth (each sector's `@prod` champion self-describes). For visualizations and downstream apps, `refresh_sector_performance()` materialises a denormalised **`sector_performance`** table in the eval DB — a refresh-only projection that joins each champion with the baseline MASE and the CBS SBI hierarchy:

| Concern | Function |
|---|---|
| Per-sector champion vs baseline tiers (Good/Medium/Poor) | `build_sector_quality_table()` |
| Enrich with SBI title + level | `enrich_with_hierarchy()` (via `m_sbi_classifier`) |
| Refresh the read-model from MLflow | `refresh_sector_performance()` |
| Read it for charts/apps | `load_sector_performance()` |
| JSON hierarchy tree (champion · model_type · sector · performance) | `to_tree()` |

Tiers are assigned on **MASE**: *Good* when MASE ≤ 0.90 (clear skill over the seasonal naive), *Medium* when 0.90 < MASE < 1.0 (beats the naive modestly), *Poor* when MASE ≥ 1.0 (no lift over the naive) or non-finite. The baseline model's MASE (`SectorQuarterRollingMean`) is carried as an informative reference column. Charts are rendered by `m_model_viz.py` (`plot_sector_leaderboard`, `plot_predicted_vs_actual`).
