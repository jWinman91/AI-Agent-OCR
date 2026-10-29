import Papa from 'papaparse'

export type AgentName = 'data_extractor' | 'analyser'
export type Workflow = 'pipeline' | 'extractor' | 'analyser'
export type OllamaDependencyStatus = {
  ollama_reachable: boolean
  required_model: string
  model_installed: boolean
  ready: boolean
  error: string | null
}
export type OllamaPullProgress = {
  status?: string
  digest?: string
  total?: number
  completed?: number
  error?: string
}
export type AgentConfig = {
  model: string
  model_provider: string
  domain_knowledge_prompt: string
  model_settings?: {
    temperature?: number
    top_p?: number
    [setting: string]: unknown
  }
  output_type?: boolean | null
}
export type AgentResult = {
  data_file_path?: string | null
  df_file_path?: string | null
  plot_path?: string | null
  code_summary?: string
  error_message?: string | null
}
export type ResultView = {
  csv?: string
  columns: string[]
  rows: string[][]
  plot?: Blob
  summary: string
  errors: string[]
}

export const outputPath = (path?: string | null) => (path && path !== '.' ? path : undefined)
const encodePath = (path: string) => path.split('/').map(encodeURIComponent).join('/')

async function request(path: string, init: RequestInit = {}, timeout = 60000): Promise<Response> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)
  try {
    const response = await fetch(path, { ...init, signal: controller.signal })
    if (!response.ok) {
      const body = await response.text()
      let detail = body
      try {
        detail = JSON.stringify(JSON.parse(body).detail ?? body)
      } catch {
        /* Plain HTTP errors remain readable. */
      }
      throw new Error(`Request failed (${response.status}): ${detail.slice(0, 500)}`)
    }
    return response
  } catch (error) {
    if (controller.signal.aborted)
      throw new Error(
        'Request timed out. Backend work may still be running; do not immediately retry.',
      )
    if (error instanceof TypeError)
      throw new Error('Cannot reach the service. Check the backend connection.')
    throw error
  } finally {
    clearTimeout(timer)
  }
}
async function json<T>(path: string, init?: RequestInit, timeout?: number): Promise<T> {
  const response = await request(path, init, timeout)
  try {
    return (await response.json()) as T
  } catch {
    throw new Error('The service returned an invalid response.')
  }
}
async function confirmed(path: string, init: RequestInit) {
  if ((await json<unknown>(path, init)) !== true)
    throw new Error('The backend did not confirm the operation.')
}
async function catalog(path: string): Promise<string[]> {
  const values = await json<unknown>(path)
  if (!Array.isArray(values) || values.some((value) => typeof value !== 'string' || !value))
    throw new Error('The backend returned an invalid model catalog.')
  return values
}
async function ollamaDependencyStatus(): Promise<OllamaDependencyStatus> {
  const value = await json<unknown>('/api/ollama/dependency-status/')
  if (
    typeof value !== 'object' ||
    value === null ||
    typeof (value as OllamaDependencyStatus).ollama_reachable !== 'boolean' ||
    typeof (value as OllamaDependencyStatus).required_model !== 'string' ||
    typeof (value as OllamaDependencyStatus).model_installed !== 'boolean' ||
    typeof (value as OllamaDependencyStatus).ready !== 'boolean' ||
    !((value as OllamaDependencyStatus).error === null || typeof (value as OllamaDependencyStatus).error === 'string')
  )
    throw new Error('The backend returned an invalid Ollama dependency status.')
  return value as OllamaDependencyStatus
}
async function pullOllamaModel(
  onProgress: (progress: OllamaPullProgress) => void,
  signal: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch('/api/ollama/pull-model/', { method: 'POST', signal })
  } catch (error) {
    if (signal.aborted) throw error
    if (error instanceof TypeError)
      throw new Error('Cannot reach the service. Check the backend connection.')
    throw error
  }
  if (!response.ok) {
    const body = await response.text()
    let detail = body
    try {
      detail = JSON.stringify(JSON.parse(body).detail ?? body)
    } catch {
      /* Plain HTTP errors remain readable. */
    }
    throw new Error(`Model download failed (${response.status}): ${detail.slice(0, 500)}`)
  }
  if (!response.body) throw new Error('The backend did not stream model download progress.')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let succeeded = false
  const consumeLine = (line: string) => {
    if (!line.trim()) return
    let progress: OllamaPullProgress
    try {
      progress = JSON.parse(line) as OllamaPullProgress
    } catch {
      throw new Error('The backend returned invalid model download progress.')
    }
    if (progress.error) throw new Error(progress.error)
    onProgress(progress)
    if (progress.status === 'success') succeeded = true
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      lines.forEach(consumeLine)
      if (done) break
    }
    if (buffer.trim()) consumeLine(buffer)
  } finally {
    reader.releaseLock()
  }
  if (!succeeded) throw new Error('Ollama ended the download without confirming success.')
}
export const api = {
  initialize: () => confirmed('/api/update_agents/', { method: 'POST' }),
  ollamaDependencyStatus,
  pullOllamaModel,
  config: (agent: AgentName) => json<AgentConfig>(`/api/get_config/?config_name=${agent}`),
  save: async (agent: AgentName, config: AgentConfig) => {
    await confirmed(`/api/upload_config/?config_name=${agent}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
    await api.initialize()
  },
  reset: async (agent: AgentName) => {
    await confirmed(`/api/reset_config/?config_name=${agent}`, { method: 'POST' })
    await api.initialize()
  },
  setKey: (key: string) =>
    confirmed(`/api/set_openai_api_key/?api_key=${encodeURIComponent(key)}`, { method: 'POST' }),
  providers: () => catalog('/api/get_model_providers/'),
  models: (provider: string) =>
    catalog(`/api/get_models/?model_provider=${encodeURIComponent(provider)}`),
  upload: async (workflow: Workflow, files: File[], resize: string) => {
    if (!files.length) throw new Error('Select a file before uploading.')
    const pdf = files.some((file) => file.name.toLowerCase().endsWith('.pdf'))
    if (workflow !== 'analyser' && pdf && files.length !== 1)
      throw new Error('Upload one PDF at a time, without additional images.')
    const allowed = workflow === 'analyser' ? /\.(csv|xlsx)$/i : pdf ? /\.pdf$/i : /\.(png|jpe?g)$/i
    if (files.some((file) => !allowed.test(file.name))) throw new Error('Unsupported file type.')
    const body = new FormData()
    files.forEach((file) => body.append(workflow === 'analyser' || pdf ? 'file' : 'files', file))
    if (workflow !== 'analyser') body.append('resize', resize)
    const path = workflow === 'analyser' ? 'uploaddatafile' : pdf ? 'uploadpdf/' : 'uploadimages/'
    if ((await json<unknown>(`/api/${path}`, { method: 'POST', body }, 300000)) !== true)
      throw new Error('Upload was not confirmed.')
  },
  run: async (workflow: Workflow, prompt: string, previousError?: string) => {
    const body = new URLSearchParams({ user_prompt: prompt })
    if (previousError) body.set('previous_error', previousError)
    if (workflow !== 'pipeline')
      body.set('agent_name', workflow === 'extractor' ? 'data_extractor' : 'analyser')
    const configured = Number(import.meta.env.VITE_AGENT_TIMEOUT_SECONDS || 1800)
    const timeout = Number.isFinite(configured) && configured > 0 ? configured * 1000 : 1800000
    const result = await json<AgentResult>(
      `/api/${workflow === 'pipeline' ? 'run_agents' : 'run_single_agent'}`,
      { method: 'POST', body },
      timeout,
    )
    if (result.error_message) throw new AgentFailure(result.error_message)
    return result
  },
}
export class AgentFailure extends Error { }

export async function loadResult(result: AgentResult, workflow: Workflow): Promise<ResultView> {
  const view: ResultView = { columns: [], rows: [], summary: result.code_summary || '', errors: [] }
  const csv = outputPath(workflow === 'extractor' ? result.data_file_path : result.df_file_path)
  const plot = outputPath(result.plot_path)
  if (!csv && !plot) view.errors.push('No plot or analysis data generated.')
  if (csv) {
    try {
      view.csv = await (
        await request(`/api/df_path/${encodePath(csv)}/download_data`, {}, 300000)
      ).text()
      const parsed = Papa.parse<string[]>(view.csv, { skipEmptyLines: 'greedy' })
      const parseError = parsed.errors.find((error) => error.code !== 'UndetectableDelimiter')
      if (parseError) throw new Error(parseError.message)
      view.columns = parsed.data[0] || []
      view.rows = parsed.data.slice(1)
    } catch (error) {
      view.errors.push(`Cannot download data: ${message(error)}`)
    }
  }
  if (plot) {
    try {
      view.plot = await (
        await request(`/api/plot_path/${encodePath(plot)}/download_plot`, {}, 300000)
      ).blob()
    } catch (error) {
      view.errors.push(`Cannot download plot: ${message(error)}`)
    }
  }
  return view
}
export const message = (error: unknown) =>
  error instanceof Error ? error.message : 'An unexpected error occurred.'
