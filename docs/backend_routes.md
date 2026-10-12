# Documentation on existing routes

## Model discovery

- `GET /get_model_providers/` returns a JSON list of provider names from
  `backend/configs/available_models.yaml`, for example `["openai", "ollama"]`.
- `GET /get_models/?model_provider=ollama` returns a JSON list of model names.
  Unknown providers return HTTP 404. Retriever failures or unregistered retriever
  names return HTTP 502 with an error in `detail`.

The catalog is loaded when the backend starts. Restart the backend after editing it.
Each entry under `model_provider` uses `retriever_function` when supplied;
otherwise its `available_models` list is returned. A retriever takes precedence
even if a static list is also configured, and errors do not fall back to that list.

To add a static provider, add an entry to the YAML:

```yaml
model_provider:
  custom:
    available_models:
      - custom-model
```

For dynamic discovery, implement a function returning `list[str]` in
`backend/src/model_retriever.py`, register it in `ModelRetriever.retrievers`, and
set the entry's `retriever_function` to its registered name. Both frontends load
their provider and model selectors from these routes. Discovery entries do not
add model execution support; the agent framework must also support the provider.
Ollama discovery uses the backend's `OLLAMA_BASE_URL` environment variable,
defaulting to `http://localhost:11434`.

## Config handling API routes:
- `/upload_config(config: ConfigModel) -> bool`:
  - Purpose: Update a config file
  - Params (ConfigModel):
      - model: str
      - model_provider: str
      - mcp_server: str
      - system_prompt_user: str
  - Return: bool (success of update)
- `/get_config(name: Literal["extractor", "plotter"]) -> ConfigModel`:
  - Purpose: Get config file by name
  - Params:
    - name of config: str
  - Return: config
- `/get_all_configs() -> List[ConfigModel]`:
  - Purpose: Get all config files
  - Return: List of all config files
- `/reset_config(name: Literal["extractor", "plotter"]) -> bool`:
  - Purpose: Resets config to the default one from config.yaml
  - Params:
    - name  of config: str
  - Return: bool (success of deletion)

## File-managing routes:
- `/uploadpdf(file: UploadFile, resize: Literal["original", "medium", "small"]) -> bool`:
  - Purpose: Upload PDF file (only one pdf allowed)
  - Params:
    - file: UploadFile
    - resize: str for resizing each page of the pdf (Form(...))
  - Return: bool (success of upload)
- `/uploadimages(files: List[UploadFile], resize: Literal["original", "medium", "small"]) -> bool`:
  - Purpose: Upload images (list of png, jpg, etc. allowed)
  - Params:
    - files: List[UploadFile]
    - resize: str for resizing each image (Form(...))
  - Return: bool (success of upload)
- `/uploaddatafile(file: UploadFile) -> bool`:
  - Purpose: Upload data file (only one csv or xlsx allowed)
  - Params:
    - file: UploadFile
  - Result: bool (success of upload)
- `/set_extern_data_source(data_source_name: Literal[...]) -> bool`:
  - Purpose: Set external data source for the AI to draw data from (e.g., yfinance, ...)
  - Params:
    - data_source_name: limited number of data sources
  - Return: bool (success of setting the data source)
- `/download_plot(plot_path: str) -> FileResponse`:
  - Purpose: Download plot by filename
  - Params:
    - plot_path: str
  - Return: FileResponse
- `/download_data(data_file_path: str) -> FileResponse`:
  - Purpose: Download data file by filename
  - Params:
    - data_file_path: str
  - Return: FileResponse

## Agent-related routes:
- `/update_all_agent() -> bool`:
  - Purpose: Recreates all agents with the current configuration from the DB
  - Return: bool (success of recreation)
- `/run_single_agent(user_prompt: str, agent_name: Literal["extractor", "plotter"]) -> AgentResult`:
  - Purpose: Single run of either extractor or plotter agent
  - Params:
    - user prompt: str, prompt coming from user
    - agent name: str, either extractor or plotter
  - Return: AgentResult
      - data_file_path: str | None
      - plot_path: str | None
      - tool_used: str | None
      - code_summary: str | None
      - error: str | None
- `/run_agent_chain(user_prompt: str) -> AgentResult`:
  - Purpose: Run the predefined chain of agent to generate the plot
  - Params:
    - user_prompt: str, prompt coming from user
  - Return: AgentResult
    - data_file_path: str | None
    - plot_path: str | None
    - tool_used: str | None
    - code_summary: str | None
    - error: str | None
