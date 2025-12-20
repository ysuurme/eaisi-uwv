# EAISI UWV

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Project Contributions](#project-contributions)
- [GitHub Flow](#github-flow)
- [Project Management](#project-management)
- [Data Management](#data-management)

## Getting Started

This project is built using CPython 3.10.11 and dependencies are managed with UV. We use GitHub as our central hub for code management and project coordination.

References:
- UV CPython installation and dependency management: https://docs.astral.sh/uv/
- Github docs: https://docs.github.com/
- Project Structure: https://github.com/ysuurme/eaisi-uwv
- Project Contributions: https://docs.github.com/en/get-started/using-github/github-flow

## Project Structure

- **Main Branch**: `main` - the stable, production-ready code
- **Feature Branches**: `feature/*` - for new features and enhancements
- **Bug Fix Branches**: `bugfix/*` - for bug fixes
- **Documentation Branches**: `docs/*` - for documentation updates

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

## Data Management
This documentation outlines the architectural strategy for our data pipeline, utilizing a Mini-Medallion Architecture implemented with Python 3, SQLAlchemy Core & ORM, and SQLite3.
The goal is to provide a lightweight, portable, and reproducible ELT (Extract, Load, Transform) framework for our machine learning project.

# 0_Raw: The "Unstructured Zone".
Technology: Local project directory (/data/0_Raw/).
Strategy: Original files (.json, .csv, etc.) exactly as received from APIs.
Excluded from Git (.gitignore) to prevent unnecessary data storage in the repository. Acts as the "Source of Truth" for debugging and re-processing.

# 1_Bronze: The "Landing Zone" and our first structured representation of the raw data.
Technology: SQLite3 database inserts via SQLAlchemy Core (optimized for high-speed bulk inserts).
Strategy: We implement a Star Schema at this level in the form of Fact (the main data) and each Dimension (the lookup tables).
- Fact Tables: Store the main numerical data (e.g., TypedDataSet).
- Dimension Tables: Store lookup data (e.g., Periods, Gender, PersonalCharacteristics).

# 2_Silver: The "Clean Zone" that focuses on data quality and integration.
Technology: SQLite3 database inserts via SQLAlchemy ORM (better for complex logic).
Strategy: Transform separate Fact and Dimension tables into a single, "meaningful" table by:
- Flattening: Extracting nested JSON structures into flat columns
- DataType Casting: Converting strings to proper DATETIME or NUMERIC types
- Standardization: Trimming whitespace (e.g., "14   " $\rightarrow$ "14") and handling NULL values
- Enrichment: Performing JOIN operations between Facts and Dimensions to replace cryptic codes (e.g., GM9001) with human-readable titles (Bonaire).

# 3_Gold: The "Business/ML Zone" that provides clean features for analysis and ML training.
Technology: SQLite3 database inserts via SQLAlchemy ORM (better for complex logic).
Strategy: Feature engineering and applying business logic by:
- Aggregations: Calculating monthly averages or regional totals.
- Feature Engineering: gold tables are structured as a Feature Store where each row represents a clean observation ready for model ingestion.
