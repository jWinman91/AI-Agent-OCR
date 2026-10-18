# AI Agent OCR Frontend

React and TypeScript workspace for the FastAPI backend. Provider and model discovery
is handled by the backend.

## Development

Start the backend from its directory, then start this frontend:

```sh
cd backend
uv run python app.py
```

In another terminal:

```sh
cd frontend
npm ci
npm run dev
```

Open the Vite URL (normally http://localhost:5173). Vite proxies `/api` to
http://127.0.0.1:8000. Set `BACKEND_URL` when starting Vite to use a different target.
Set `OLLAMA_BASE_URL` on the backend to change its Ollama discovery target.
Set `VITE_AGENT_TIMEOUT_SECONDS` to change the default 1800-second agent timeout.
A browser timeout does not cancel backend work; avoid submitting duplicate runs.

## Workflows

- Configure the data extractor and analyser: provider/model, user system prompt,
  OpenAI key, save and reset. Providers and models come from the backend catalog
  in `backend/configs/available_models.yaml`; Ollama models are discovered dynamically.
  Keys are not stored in browser storage.
- Run the full pipeline with one multi-page PDF or multiple PNG/JPEG files.
- Run just the extractor (five-row result preview) or analyser (CSV/XLSX input,
  or an external-data request without upload).
- Dataset selection uploads immediately. Image and PDF selection opens a resolution
  dialog; confirming starts the upload, and cancelling leaves earlier uploads intact.
  File selection and drag-and-drop use the same validation and upload flow.
- Run views place files on the left and a prominent prompt on the right (stacked on
  mobile). The single-agent selector appears above the heading. Running feedback
  shows an activity indicator and elapsed time until generated outputs finish loading.
  Results appear below and scroll into view once per completed run. Analyser tables
  and plots appear side-by-side on desktop, or use full width when only one exists.
- Resize settings are submitted in upload form data. Generated CSVs have filtering,
  sorting, pagination and downloads; plots display the backend-generated image.
  The analyser never redisplays the input CSV as its result.
- Upload markers are consumed after each run attempt. Results, prompts and retry
  errors remain in memory across navigation. Refreshing the browser resets that state.

The backend uses shared upload state. Multiple browser sessions connected to the same
backend can replace another workflow's files. Use one active session at a time.

## Branding

Change semantic colors, font family, borders and radius in `src/theme.css`.
Layout and component styles are in `src/App.css`. The bundled Public Sans font
does not require an external font service. Accessible tabs use Radix primitives;
buttons/forms consume the same theme tokens. Brand assets can be added under `src/assets`.

## Verification

```sh
npm test
npm run build
npm run lint
npm exec playwright install chromium
npm run test:e2e
```

Unit/component and desktop/mobile browser tests mock backend responses and do not
send model requests. Browser screenshots are saved under ignored `test-results/`.
TanStack Table produces a React Compiler compatibility warning; the compiler is
not enabled and table behavior is covered by tests.

## Production

`npm run build` generates static files in `dist/`. Serve these with SPA fallback
to `index.html` and a reverse proxy for `/api/*` (strip `/api`).
Configure long proxy timeouts for agent calls. Vite's development proxy is not bundled
into the production build.