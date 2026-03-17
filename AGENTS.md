# Repository Guidelines

## Project Structure & Module Organization
`bitoguard_core/` contains the main Python backend: FastAPI endpoints in `api/`, data access in `db/`, feature builders in `features/`, batch pipelines in `pipeline/`, model code in `models/`, production helpers in `services/`, and pytest suites in `tests/`. Competition-aligned pipelines and experiments live in `bitoguard_core/official/` and `bitoguard_core/transductive_v1/`. Repo-tracked data products, reports, and model bundles live in `bitoguard_core/artifacts/`.

`bitoguard_frontend/` is a Next.js App Router app with routes in `src/app/`, shared UI in `src/components/`, and static assets in `public/`. `bitoguard_mock_api/` provides a read-only FastAPI mock of the upstream source API backed by `bitoguard_sim_output/`. Sample and simulator assets live in `bitoguard_sample_output/`, `bitoguard_sim_output/`, `bitoguard_simulator/`, and `data/aws_event/`. AWS infrastructure lives in `infra/aws/terraform/` and `infra/aws/lambda/`; runbooks, specs, and deployment notes live in `docs/` and `deploy/`.

## Build, Test, and Development Commands
- `make setup`: create `bitoguard_core/.venv` and install backend dependencies.
- `make test` or `make test-quick`: run the backend pytest suite.
- `make sync`, `make features`, `make features-v2`, `make train`, `make score`, `make drift`, `make refresh`: run the core backend data and modeling pipeline from the repo root.
- `make serve`: start FastAPI on `http://localhost:8001`.
- `make frontend`: start the Next.js app on `http://localhost:3000`.
- `cd bitoguard_frontend && npm ci && npm run dev`: install frontend deps and start Next.js on `http://localhost:3000`.
- `cd bitoguard_frontend && npm run lint && npm run build`: required frontend validation.
- `cd bitoguard_mock_api && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pytest`: validate the mock API when its behavior changes.
- `docker compose up --build`: run the full stack locally.
- `cd infra/aws/terraform && terraform fmt -check -recursive && terraform validate`: required for Terraform changes.

## Coding Style & Naming Conventions
Python uses 4-space indentation, type hints, and `snake_case` for functions, modules, and tests. Keep backend and mock-API code domain-focused and colocated with the existing package structure. Frontend code uses strict TypeScript, React function components, `PascalCase` component files such as `Sidebar.tsx`, and route files like `src/app/alerts/page.tsx`. Follow the existing frontend style: double quotes, no semicolons, and Tailwind utility classes.

## Testing Guidelines
Add or update pytest cases for every backend behavior change in `bitoguard_core/`, and update `bitoguard_mock_api/tests/` when mock API contracts change. Name files `test_*.py` and prefer focused fixture-driven cases using `bitoguard_core/tests/conftest.py` where applicable. Official and transductive pipeline changes should keep artifact-free coverage in `bitoguard_core/tests/`. There is no dedicated frontend test harness yet, so treat `npm run lint` and `npm run build` as the minimum gate for UI work.

## Commit & Pull Request Guidelines
Recent history follows Conventional Commit prefixes: `feat:`, `fix:`, `chore:`, and `docs:`. Keep subjects imperative and scoped to one change. PRs should include a short summary, linked issue or context, exact validation commands run, and screenshots for UI changes. For Docker or infra changes, note required env vars such as `BITOGUARD_API_KEY` and confirm `docker compose build` or Terraform validation passed.

## Security & Configuration Tips
Start from `deploy/.env.compose.example` for Docker and `bitoguard_frontend/.env.example` for frontend configuration. Do not commit secrets, local tool state, or ad hoc database files; only commit refreshed artifacts or dataset snapshots when they are intentional repo outputs. Preserve `BITOGUARD_GRAPH_FEATURES_TRUSTED_ONLY=true` unless you are intentionally changing the graph trust boundary and updating the corresponding docs.
