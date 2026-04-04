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

**To run the Machine Learning lifecycle only:**
Trigger the "Train $\rightarrow$ Evaluate $\rightarrow$ Register" lifecycle directly on pre-existing gold tables.
```bash
# Default: 80072ned_gold, RandomForest, features=ALL
uv run main.py  
```

**To refresh the underlying Data Pipeline *before* ML execution:**
Append the `--refresh-data` flag to natively execute the full Medallion architecture (Raw $\rightarrow$ Bronze $\rightarrow$ Silver $\rightarrow$ Gold) ensuring all enabled `config.py` metrics are perfectly synced before the Machine Learning process takes over.
```bash
uv run main.py --refresh-data
```

### 3. Track Results (MLflow)
The pipeline is "Zero-Artifact"; all results are stored in `data/4_eval/eval_data.db`. Launch the UI to view metrics, tuning grids, and model signatures.
🌐 **Open**: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 4. Run Tests
Execute all unit and end-to-end tests across the project to verify everything is working correctly:
```bash
uv run python -m unittest discover -s tests
```

---
**Note**: To auto-launch the UI during training, set `START_MLFLOW_UI = True` in `config.py`.

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
│   ├── ml_engineering/         # ML Lifecycle (Training, Evaluation, Registry, Orchestrator)
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

## Machine Learning Engineering
This documentation outlines the architectural strategy for our machine learning operations. We leverage **MLFlow** for experiment tracking, metric logging, and managing the model registry to ensure a robust MLOps lifecycle.

**Experimentation**
- Technology: Jupyter (.ipynb) notebooks integrated with Git and MLFlow.
- Strategy: Collaborative exploratory data analysis and prototyping with tracked experiments for reproducibility of data, hyperparameters, and metrics.

**Model Training**
- Technology: Scikit-learn and PyTorch frameworks with optional support for XGBoost, LightGBM, Keras and TensorFlow.
- Strategy: Scalable model development enabling efficient hyperparameter tuning, target optimization, and automated feature engineering.

**Model Evaluation**
- Technology: MLFlow for tracking and visualization, combined with evaluation datasets.
- Strategy: Interactive and automated assessment of model effectiveness, including performance comparison, bias detection, and explainability analysis.

**Model Registry**
- Technology: Centralized MLFlow Model Registry through mlflow.log_artifact("model.pkl")
- Strategy: Govern the full model lifecycle including versioning, metadata storage, documentation, and release management.

**ML Pipelines**
- Technology: Python-based orchestration and pipeline objects.
- Strategy: Automate complex training and prediction workflows, triggering pipelines on-demand or on-schedule while capturing execution metadata.