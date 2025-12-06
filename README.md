# EAISI UWV

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Project Contributions](#project-contributions)
- [GitHub Flow](#github-flow)
- [Project Management](#project-management)
- [Contributing](#contributing)

## Getting Started

This project is built using CPython 3.10.11 and managed with UV. We use GitHub as our central hub for code management and project coordination.

References:
Python installation and dependency management via UV: https://docs.astral.sh/uv/
Project code management via Github: https://docs.github.com/
Project repository: https://github.com/ysuurme/eaisi-uwv
Project contributions: https://docs.github.com/en/get-started/using-github/github-flow

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

Our development process follows the GitHub Flow model for a smooth and collaborative workflow.
: https://docs.github.com/en/get-started/using-github/github-flow

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

We use **GitHub Projects** to organize our work, track progress, and plan issues.

### How We Use GitHub Projects

- **Issue Organization**: Features, bugs, and documentation are tracked as issues and organized in @project-eaisi-uwv
- **Status Tracking**: Issues move through columns: `Todo` → `In Progress` → `In Review` → `Done`
- **Priority & Assignment**: Issues are prioritized and assigned to team members

### Getting Started with Issues

1. Check the current **GitHub Project** board for available issues
2. Find a task in the `Backlog` that interests you
3. Assign yourself to the issue
4. Move it to `In Progress` when you start working
5. Create a PR linked to the issue
6. Move to `In Review` when the pull request is ready
7. Move to `Done` after the Pull Request is merged to main

