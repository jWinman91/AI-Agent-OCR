# AI-Agent-OCR

## Run with uv
- Install uv (macOS):

```bash
brew install uv
# or
curl -Ls https://astral.sh/uv/install.sh | sh
```

- Setup the environment and lock dependencies:

```bash
uv lock
uv sync
```

- Run the FastAPI backend:

```bash
uv run python backend/app.py
```

Notes:
- This project targets Python 3.11 (see `mypy.ini`).
- Dependencies are declared in `pyproject.toml`; `requirements.txt` remains for reference.

## React frontend

The brandable React/TypeScript frontend lives in [frontend](frontend/README.md).
With the backend running, start the frontend:

```bash
cd frontend
npm ci
npm run dev
```

Open http://localhost:5173. See [frontend/README.md](frontend/README.md) for theme
tokens, API proxy settings, tests, and static deployment requirements.

## Pre-commit checks

Install the development tools and Git hooks once per clone:

```bash
uv sync --extra dev
uv run --extra dev pre-commit install
```

Commits check staged Python files with Ruff. Changes to backend Python code or
typing configuration also run strict mypy checks on all of `backend/src`.
Backend and frontend test suites run on every commit in separate Python processes.
The tests use mocks and temporary files; no running models are required.
The hooks require `uv` on your PATH and do not automatically modify files.
Changes under `frontend/` also run its build, unit tests and lint checks;
run `npm ci` inside that directory first. These hooks require Node.js and npm.

Run all checks across the repository manually:

```bash
uv run --extra dev pre-commit run --all-files
```

## Frontend verification

Run the frontend regression tests (mocked HTTP; no models required):

```bash
npm --prefix frontend test
```

Agent requests have a 30-minute read timeout by default. A frontend timeout does not
cancel backend work; avoid immediately submitting the same request again.