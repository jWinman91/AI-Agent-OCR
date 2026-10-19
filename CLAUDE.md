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
mypy --config-file evaluation/mypy.ini evaluation/evaluate_analyser.py evaluation/evaluate_extractor.py evaluation/src evaluation/tests
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
The `evaluation/` directory is separate from `backend/` and `frontend/`, with distinct entry points and source packages for the data extractor and analyser evaluations.

- `src/experiments.py` holds everything shared by both evaluators: path constants, prompt loading, experiment discovery (folders for the extractor; the analyser's experiments are its `prompt.yaml` entries), readiness checks, the run-all loop that skips experiments with missing data (`ExperimentDataMissing`), metric flattening and MLflow logging. `src/backend_client/agent_client.py`'s `AgentClient` wraps the backend calls both evaluators need (update agents, upload images/PDF/data file, run a single agent, get config, download data/plot) on top of the low-level `request_be.py`/`response_be.py`; it returns agent results unchecked, so each evaluator decides whether an `error_message` aborts the run.
- All paths are resolved relative to `evaluation/` (not the current working directory), so the CLIs can be started from anywhere. Test data is split per agent: `eval_data/extractor/` and `eval_data/analyser/`, each with its own `prompt.yaml` keyed by experiment name. `ground_truth/` stays one shared directory.
- MLflow logs to `evaluation/mlflow.db`, with artifacts of newly created experiments under `evaluation/mlruns/`; view with `mlflow ui --backend-store-uri sqlite:///mlflow.db` from `evaluation/`. MLflow experiment = experiment name (prefixed `analyser/` for the analyser, so it never collides with an extractor experiment of the same name), MLflow run = `--run_name`. Missing values are logged as NaN.
- Tests: `uv run --extra dev python -m unittest discover -s evaluation/tests -t evaluation` (no backend or API key needed; judge calls use a pydantic_ai `FunctionModel`, MLflow a temp sqlite store).

#### Extractor evaluation
- `evaluate_extractor.py`, implementation in `src/extractor/` (`evaluate.py`, `extract_data.py`). Example: `python evaluate_extractor.py --experiment_name body_weight --run_name gemma4_v1`, or `--run_name` alone to run every extractor experiment.
- Each experiment is a folder `eval_data/extractor/{experiment}/` of images/PDF, scored against `ground_truth/{experiment}.csv`. Output goes to `extracted_data/` (gitignored).
- Compares extracted CSV data against hand-labeled ground truth: matches field values per source image, normalizes numeric and currency values, reports classification metrics for multi-value fields and MAE for single-value fields, and writes mismatch CSVs as temporary files for MLflow artifacts.

#### Analyser evaluation (LLM-as-a-judge)
- `evaluate_analyser.py`, implementation in `src/analyser/`: `experiment_spec.py` (prompt + ground truth lookup), `run_analyser.py` (upload data file → `/run_single_agent` → download plot/output data), `judge.py` (`LLMJudge`, pydantic_ai structured output), `judge_config.py` + `judge_config.yaml` (judge model, prompts, 1–5 rubrics), `evaluate.py` (`AnalyserEvaluator`, gating + metrics).
- Every top-level key of `eval_data/analyser/prompt.yaml` is one experiment, mapping to `data_file` (CSV/Excel in `eval_data/analyser/`, may be a symlink into `ground_truth/`), `prompt` and an optional `ground_truth` (file in `ground_truth/`); several experiments can share a data file. Run-all runs every entry. A named ground truth must exist and is a `.png` reference plot or a `.csv` expected output; it is shown to the judge as a reference.
- Metrics: `execution_success` (1/0 from the backend's `error_message`; the code is never re-executed), `code_simplicity` (judged whenever code exists, even on failure), and `data_correctness`, `task_relevance`, `chart_suitability`, `completeness`, `readability` (judged only when execution succeeded and a plot exists, otherwise NaN). The visualisation judge sees the plot, request, input data and reference but deliberately not the code. A failing judge call leaves its scores NaN and sets a `judge_error` tag instead of aborting.
- The judge (`openai:gpt-6.1-sol` by default) needs `OPENAI_API_KEY` in the environment; the CLI checks the key and makes one preflight call before running the analyser. Params include `judge_config_sha`, so only compare scores between runs judged with the same `judge_config.yaml`. Artifacts: plot, output data, code, code summary, `judge_results.json` (explanations/evidence), errors and both configs.
