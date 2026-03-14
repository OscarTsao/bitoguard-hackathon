# VS Code Workflow Baseline

## What is configured

1. `.vscode/settings.json` for Python, TypeScript, linting, and quality-of-life defaults.
2. `.vscode/extensions.json` with recommended and avoided extensions.
3. `.vscode/tasks.json` for setup, test, lint, run, and pipeline commands.
4. `.vscode/launch.json` for backend, frontend, and full-stack debugging.

## First-run checklist

1. Open the workspace root (`bitoguard-hackathon`).
2. Install recommended extensions when prompted.
3. Run task: `Setup: Python venv + frontend deps`.
4. Select Python interpreter: `bitoguard_core/.venv/bin/python`.
5. Run `Run: Full stack (local)` task.

## Fast commands

1. `Tasks: Run Task` -> `Tests: quick`
2. `Tasks: Run Task` -> `Lint: backend + frontend`
3. `Run and Debug` -> `Debug: Full stack`
