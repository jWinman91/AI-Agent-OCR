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

- Run the Streamlit app:

```bash
uv run streamlit run streamlit-app/main.py
```

Notes:
- This project targets Python 3.11 (see `mypy.ini`).
- Dependencies are declared in `pyproject.toml`; `requirements.txt` remains for reference.