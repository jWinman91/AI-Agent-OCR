# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

- **Ollama** must be installed and running locally — the backend calls local models (MiniCPM-o, OpenThinker, Qwen3, Qwen3-coder) through Ollama by default. Pull the models referenced in `backend/configs/agents.yaml` before running the pipeline.
- OpenAI can be used instead of Ollama as an alternative model provider.

## Commands

```bash
# Install dependencies
uv lock && uv sync

# Run backend (FastAPI, port 8000)
uv run python backend/app.py

# Run frontend (Vite development server)
cd frontend && npm run dev

# Lint
ruff check .

# Type check
mypy backend/src --config-file mypy.ini
```

## Architecture

**AI-Agent-OCR** is a multi-agent tool that extracts data from photos (e.g. bathroom scales, utility meters) and PDF documents, then turns the extracted readings into visualizations such as time series plots of body weight. Users upload images or PDFs → agents extract structured data → optional analysis/plotting. The system runs locally using Ollama (default) or OpenAI, with `pydantic_ai` as the agent framework used to call the models.

### Application components
- **Backend** (`backend/app.py`): FastAPI server that manages agents, files, and orchestration
- **Frontend** (`frontend/`): React and TypeScript UI that calls the backend via HTTP
  - Styled to the winmor brand (winmor.ai): colour/font tokens in `frontend/src/theme.css` (navy `#14213D`, red `#E63946`, gold `#F2B134`, Manrope + Source Sans 3 via `@fontsource`), isotype SVGs in `frontend/src/assets/brand/` and favicon/app icon in `frontend/public/`, all copied from `winmor-brand/` (which also holds the brand book).

### Agent pipeline

Agents are defined in `backend/configs/agents.yaml` and instantiated by `AgentsFactory` (`backend/src/agents/agents_factory.py`). The pipeline is coordinated by `AgentRunner` (`backend/src/agents/agents_runner.py`):

1. **Orchestrator** (OpenThinker) — interprets user request, produces prompts for sub-agents
2. **Data Extractor** (MiniCPM-o-2.6, vision) — loops over images, returns structured JSON
3. **Analyser** (Qwen3-coder) — runs analysis/plotting via MCP tools
4. **Data Downloader** (Qwen3) — fetches external data (yfinance, HTTP)
5. **Fix JSON** (OpenThinker) — fallback agent to repair malformed JSON output

Agent outputs are validated by `OutputValidator` (`backend/src/agents/output_validator.py`), which tries to extract JSON from markdown blocks and falls back to the fix_json agent.

#### Data extractor output format
`DataExtractorResponse.data` is always a `list[dict[str, Any]]` — one dict per extracted row, regardless of whether one or many rows are extracted (e.g. `[{"body_weight": 70.5}]` for a single reading, `[{"body_weight": 70.5}, {"body_weight": 71.0}]` for multiple). The orchestrator's `json_details` only ever describes the per-field types of a single row (`"string"`, `"float"`, `"int"` — never `list[...]`); repetition across rows is expressed by returning multiple row objects, not list-valued fields. `AgentRunner.run_data_extractor` (`backend/src/agents/agents_runner.py`) flattens each image's row objects directly into the extracted DataFrame, attaching `timestamp` and `image_url` to every row.

For PDF pages, the data extractor receives both a page image and the same page's text (extracted with pymupdf4llm) as two representations of one page, not two separate pages. `AgentsFactory.run_agent` (`backend/src/agents/agents_factory.py`) wraps that text between `<EXTRACTED PDF TEXT>` and `</EXTRACTED PDF TEXT>` tags in the user message so the model can locate it unambiguously. The `data_extractor` system prompt in `backend/configs/agents.yaml` references these tags explicitly and warns against double-counting a value that appears both in the image and inside the tagged text block, asking the model to check any identical row objects before finalizing output and keep duplicates only if justified as two distinct occurrences on the page (e.g. two separate table rows). This is instructive rather than enforced, since gemma4 (the default local model) is small enough that overly rigid rules tend to produce worse results than clear, example-driven guidance.

As a hard backstop, `AgentRunner.run_data_extractor` (`backend/src/agents/agents_runner.py`) still enforces two things in code regardless of what the model outputs: `_deduplicate_records` drops row objects that are exact duplicates (all fields identical) before they're added to the extracted data, and `_strip_linebreaks` replaces embedded newlines in string field values with spaces so a multi-line OCR value can't corrupt the exported CSV's row structure.

### MCP servers (subprocesses)
- `backend/src/server/plot_mcp_server.py` — tools for Python code execution, dataframe inspection, pip installs
- `backend/src/server/download_data_mcp_server.py` — tools for yfinance and proxied HTTP requests

Agents that need tools spawn their MCP server as a subprocess via `pydantic_ai.mcp.MCPServerStdio`.

### Configuration system
Agent configs live in `backend/configs/agents.yaml` and are loaded into a SQLite database (`config.db`) on startup via `ConfigBuilder` (`backend/src/config_builder.py`). Users can edit agent system prompts via the UI, which are persisted to the DB. System prompts use `$user_part` as a template placeholder for the user-editable section.

### Data models
All agent response shapes and config models are in `backend/src/utils/data_models.py`. Key models:
- `ConfigModelBuild` — compiled config passed to `AgentsFactory`
- `OrchestratorResponse`, `DataExtractorResponse`, `AnalyserResponse`, `DataDownloadResponse` — typed agent outputs
- `AgentResult` — final result (plot path, data file path, code summary)

### Key FastAPI routes
- `POST /run_agents` — full pipeline
- `POST /run_single_agent` — single agent (data_extractor or analyser)
- `POST /uploadimages/` — upload + resize images (preprocessor handles EXIF, PDF→PNG)
- `POST /uploadpdf/` — converts a PDF to per-page images (`Preprocessor.convert_pdf_to_images`) and per-page text files (`Preprocessor.convert_pdf_to_text`, `.txt` extension, plain text via `pymupdf4llm.to_text` with OCR fallback), uploading both so the data extractor receives image + text pairs per page
- `POST /upload_config/` / `GET /get_config/` / `POST /reset_config/` — config CRUD

### Linting/typing config
- `ruff.toml`: line-length 88, double quotes
- `mypy.ini`: Python 3.11, strict mode, targets `backend/src`

### Evaluation harness (`evaluation/`)
The `evaluation/` directory is separate from `backend/` and `frontend/`, with distinct entry points and source packages for the data extractor and analyser evaluations. The analyser evaluation is an empty scaffold for now; no analyser scoring or MLflow integration is implemented yet.

- `evaluate_extractor.py` is the existing data extractor evaluation CLI. Its extractor-specific implementation lives in `src/extractor/` (`evaluate.py` and `extract_data.py`). `src/experiments.py` holds everything shared by both evaluators: path constants, prompt loading, experiment discovery and readiness checks, the run-all loop that skips experiments with missing data, metric flattening and MLflow logging. `src/backend_client/agent_client.py`'s `AgentClient` wraps the backend calls both evaluators need (update agents, upload images/PDF/data file, run a single agent, get config, download data/plot) on top of the low-level `request_be.py`/`response_be.py`; it returns agent results unchecked, so each evaluator decides whether an `error_message` aborts the run.
- `evaluate_analyser.py` is the analyser evaluation entry point, with its implementation package under `src/analyser/`.
- Example: `python evaluate_extractor.py --experiment_name body_weight --run_name gemma4_v1`, or `--run_name` alone to run every extractor experiment. All paths are resolved relative to `evaluation/` (not the current working directory), so it can be started from anywhere.
- Test data is split per agent so "run all" only picks up that agent's experiments: `test_data/extractor/{experiment}/` and `test_data/analyser/{experiment}/`, each agent with its own `prompt.yaml` keyed by experiment name. `ground_truth/` stays one shared directory (`ground_truth/{experiment}.*`). Extractor output goes to `extracted_data/`. `test_data/` and `extracted_data/` are gitignored.
- Extractor evaluation compares extracted CSV data against hand-labeled ground truth. It matches field values per source image, normalizes numeric and currency values, reports classification metrics for multi-value fields and MAE for single-value fields, and writes mismatch CSVs as temporary files for MLflow artifacts.
- Extractor metrics, params, and artifacts are logged to `sqlite:///mlflow.db`; run `mlflow ui --backend-store-uri sqlite:///mlflow.db` from `evaluation/` to view results.
