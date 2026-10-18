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

### Evaluation harness (`evaluate_extractor/`)
A sibling directory (outside `backend/` and `frontend/`) that evaluates `data_extractor` agent accuracy against hand-labeled ground truth, driven by `evaluate_extractor/evaluate_extractor.py` as a CLI tool.

- `src/backend_client/` (`request_be.py`, `response_be.py`) is a thin HTTP client for uploading files, running agents, and fetching results from the FastAPI backend, plus `ollama_retriever.py` for listing locally available Ollama models. File uploads pass `(filename, file)` tuples to `requests`.
- `src/extract_data.py`'s `DataExtractor` uploads test files (PDF-first routing: a single `.pdf` goes to `/uploadpdf`, otherwise all files go to `/uploadimages`), runs `/run_single_agent` with `agent_name=data_extractor`, and saves the resulting CSV to `extracted_data/{experiment_name}_{run_name}.csv`. The extracted CSV is the only output kept permanently on disk; the agent's config YAML (`save_config`) is written to a temp file (system temp dir) and returned to the caller so it can be logged as an MLflow artifact and then deleted. `evaluate.py`'s `save_false_values` works the same way (temp files, returned paths, caller deletes after upload).
- `src/evaluate.py`'s `DataExtractorEvaluator` compares `extracted_data/{experiment_name}_{run_name}.csv` against `ground_truth/{experiment_name}.csv` (ground truth is `;`-separated). Matching is **per field**: each ground truth column (except the `image_url`/`timestamp` metadata columns) is scored independently. Extracted and ground truth values for one field are grouped by `image_url` (must match exactly) and bag-matched (unordered multiset) within each group, so order on the page doesn't matter and a correct value for one field still counts even if another field in that row was wrong. Column names are matched case-insensitively (e.g. ground truth `Saldo` ↔ extracted `SALDO`). Values are normalized before comparison via `_normalize_value`/`_parse_numeric`, which strips currency symbols (`€`/`$`/`£`) and handles both German/European (`"60.394,59 €"`) and US (`"1,234.56"`) thousands/decimal separator conventions so differently-formatted equal numbers still match.
- `evaluate_extracted_data` returns `{"fields": {field_name: {accuracy, precision, recall, f1_score}, ...}, "overall": {...}}`: one metric set per ground truth field, plus a macro-averaged `"overall"` set. A ground truth field the extractor was never asked to produce (no matching extracted column) has all four metrics set to `None` and is excluded from the macro average; a field that was attempted but scored 0.0 is a real result and still counts toward the average. `save_false_values` writes a `{experiment_name}_{run_name}_false_{positives,negatives}.csv` pair (temp files) with **one row per (image_url, field) page that has a mismatch**: the false positives file has an `image_url, field, hallucinated_values` schema (extracted values not in that page's ground truth, as a list), and the false negatives file has `image_url, field, missed_values` (ground truth values for that page that were never extracted, as a list). Pages with no mismatch produce no row.
- All experiment prompts live in one shared `test_data/prompt.yaml`, keyed by experiment name (not one `prompt.yaml` per experiment subfolder), so they stay easy to see/edit together.
- `evaluate_extractor.py --experiment_name <name> --run_name <name>` runs one experiment end-to-end (upload → extract → save → evaluate → log to MLflow) and raises immediately on any error. If `--experiment_name` is omitted, every subdirectory of `test_data/` is run in turn via `run_all_experiments`; an experiment missing its `prompt.yaml` entry, its `ground_truth/{name}.csv`, or its test files (checked upfront by `check_experiment_ready`, before any backend call) is logged as an error and skipped rather than aborting the batch — only this "missing data" case is caught, any other failure (backend/extraction/evaluation errors) still stops the run.
- Metrics, params (`user_prompt`, `model`, `model_provider`), and the output files (extracted CSV + config YAML + the two false-value CSVs) are logged to MLflow (`sqlite:///mlflow.db`, tracked via `mlflow.set_tracking_uri` at module load — run the CLI from `evaluate_extractor/` so this resolves correctly) with a 1:1 mapping: MLflow experiment = `experiment_name`, MLflow run = `run_name`, so repeated runs of the same experiment accumulate under one MLflow experiment. The nested metrics dict is flattened into MLflow's flat namespace by `flatten_metrics` as `{metric}_{field}` and `{metric}_overall` (e.g. `accuracy_Saldo`, `f1_score_overall`); a `None` field metric is logged as `NaN` (MLflow rejects `None` but accepts `NaN` as its native "missing value", rendered blank in the UI). Skipped experiments get an MLflow run with only the `*_overall` metrics set to `NaN` plus a `skipped_reason` tag. View results with `mlflow ui --backend-store-uri sqlite:///mlflow.db` from `evaluate_extractor/`.
