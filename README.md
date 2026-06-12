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
| `model_key` | `linear` | Key from the ModelConfiguration catalog |
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

# Linear model, skip SBI arg, use specific feature groups
uv run main.py master_data_ml_preprocessed linear - compensation,working_conditions

# Refresh underlying data pipeline before ML execution
uv run main.py --refresh-data
```

**Available model keys:** `baseline`, `linear`, `random_forest`, `gradient_boosting`, `hist_gradient_boosting`

### 3. Track Results (MLflow)
The pipeline is "Zero-Artifact"; all results are stored in `data/4_eval/eval_data.db`. Launch the UI to view metrics, tuning grids, and model signatures.

```bash
uv run mlflow ui --backend-store-uri sqlite:///data/4_eval/eval_data.db
```

🌐 **Open**: [http://127.0.0.1:5000](http://127.0.0.1:5000)

All runs land in the `master_SickLeave_4Q` experiment, tagged with `sector` and `forecast_horizon=4Q` for easy cross-sector comparison. **MAPE** is the primary metric and quality gate; each sector has one registered model whose `@prod` alias is its current champion (the lowest-MAPE model seen so far).

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
├── data/                       # Data storage (raw, medallion db's and MLFlow registry db)
├── docs/                       # Documentation
├── models/                     # Model artifacts
├── notebooks/                  # Non-production experimentation
│   ├── data_exploration/       # Data exploration and schema analysis
│   └── ml_experimentation/     # ML experimentation and baseline models
├── src/                        # Production-ready source code
│   ├── data_engineering/       # Data Medallion Pipeline (Raw, Bronze, Silver, Gold)
│   ├── ml_engineering/         # ML Lifecycle (Modular 6-Step Pipeline)
│   │   ├── ml_orchestrator.py  # Pipeline Hub (Step 0) + run_sector_sweep()
│   │   ├── ml_1_data_extraction.py
│   │   ├── ml_2_data_validation.py
│   │   ├── ml_3_data_preparation.py
│   │   ├── ml_4_model_training.py
│   │   ├── ml_5_model_evaluation.py
│   │   ├── ml_6_model_validation.py
│   │   └── model_configs.py    # Estimator Catalog, ORM Definitions, SectorQuarterRollingMean
│   ├── utils/                  # Shared utilities and database handlers
│   └── config.py               # Project-wide configuration
├── main.py                     # Entry point for product orchestration
├── pyproject.toml              # Dependency management (UV)
└── README.md                   # Project overview
```

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
- The test set is always the last 4 quarters (1 year = the forecast horizon)
- The baseline and all competing models are evaluated on held-out future quarters
- Hyperparameter tuning uses `ExpandingWindowSplitter(fh=[1,2,3,4], step_length=4, initial_window=40)` — one full year per CV fold, evaluating all 4 ahead-steps simultaneously

### Operational Modes
Every pipeline run operates on exactly one quarterly time series (1 row per quarter):

| Mode | How | Series used |
|------|-----|-------------|
| **All-industry** (default) | Filters `BedrijfskenmerkenSBI2008_T001081 == 1` | CBS national total |
| **Sector-specific** | Filters `BedrijfskenmerkenSBI2008_<code> == 1` | One SBI sector |
| **Sector sweep** | `--all-sectors` flag | All 39+ OHE columns, one run each |

### Estimator Catalog (`model_configs.py`)

| Key | Estimator | Tunable |
|-----|-----------|---------|
| `baseline` | `SectorQuarterRollingMean(n_years=3)` | No |
| `linear` | `make_reduction(LinearRegression, window_length=12)` | `window_length` |
| `random_forest` | `make_reduction(RandomForestRegressor, window_length=12)` | `window_length`, `n_estimators`, `max_depth` |
| `gradient_boosting` | `make_reduction(GradientBoostingRegressor, window_length=12)` | `window_length`, `n_estimators`, `learning_rate` |
| `hist_gradient_boosting` | `make_reduction(HistGradientBoostingRegressor, window_length=12)` | `window_length`, `learning_rate`, `max_iter` |

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
- Temporal train/test split: last `n_test=4` quarters (1 year) held out as test set.
- Sets a `DatetimeIndex` for sktime compatibility.
- Outputs `X_train, X_test, y_train, y_test` + lineage metadata.

**4. Model Training (`ml_4_model_training.py`)**
- **Baseline path** (`SectorQuarterRollingMean`): standard sklearn `.fit(X, y)`. Logged via `mlflow.sklearn.log_model`.
- **sktime path** (all others): `.fit(y=y_train, X=X_numeric)`. Tuned via `ForecastingGridSearchCV` + `ExpandingWindowSplitter`. Serialized as pickle artifact.
- Logs `model_class`, `train_rows`, `feature_count`, `sector`, `forecast_horizon` to MLflow.

**5. Model Evaluation (`ml_5_model_evaluation.py`)**
- Walk-forward (rolling-origin) evaluation with nested inner/outer folds; headline metrics are computed on the honest **outer** folds.
- Computes **MAPE** (primary), R², MAE, RMSE on the 4-quarter-ahead forecasts and logs them to MLflow. Per-row predictions are persisted for cross-model comparison.

**6. Model Validation & Registry (`ml_6_model_validation.py`)**
- **MAPE champion/challenger gate**: one registered model per sector (sector-keyed name). A challenger is promoted to `@prod` only if its MAPE is finite and strictly lower than the incumbent champion's MAPE — or seeded unconditionally when no champion exists yet. Losing runs stay diagnosable (`passed_gate=false`) but are not registered, so the registry only grows on genuine improvement.
- R² is recorded alongside (optional `r2_floor`, disabled by default). The promoted version is tagged with `mape` / `r2` / `model_family`, so each registered model's accuracy is readable straight from the MLflow UI.
