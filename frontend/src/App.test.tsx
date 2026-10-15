import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import App from './App'

beforeEach(() => {
  vi.mocked(Element.prototype.scrollIntoView).mockClear()
  window.history.replaceState({}, '', '/single')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.includes('ollama/dependency-status'))
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: true,
          ready: true,
          error: null,
        }))
      if (url.includes('get_config'))
        return new Response(
          JSON.stringify({
            model: 'gpt-5',
            model_provider: 'openai',
            model_settings: { temperature: 0.01, top_p: 1 },
            system_prompt_user: 'Extract accurately.',
          }),
        )
      if (url.includes('get_model_providers'))
        return new Response(JSON.stringify(['openai', 'ollama', 'custom']))
      if (url.includes('get_models'))
        return new Response(JSON.stringify(
          url.includes('ollama') ? ['gemma4:latest'] : url.includes('custom') ? ['custom-model'] : ['gpt-5'],
        ))
      if (url.includes('run_single_agent'))
        return new Response(
          JSON.stringify({
            df_file_path: 'plots/output.csv',
            plot_path: null,
            code_summary: 'Monthly totals',
          }),
        )
      if (url.includes('download_data')) return new Response('Month,Total\nJanuary,123.45\n')
      return new Response('true')
    }),
  )
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

async function renderReadyWorkspace() {
  render(<App />)
  await screen.findByText('Backend connected')
}

it('keeps the workspace blocked when model installation is declined', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.includes('ollama/dependency-status'))
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: false,
          ready: false,
          error: null,
        }))
      return new Response('true')
    }),
  )
  render(<App />)
  expect(await screen.findByRole('dialog', { name: 'Download required model?' })).toBeVisible()
  fireEvent.click(screen.getByRole('button', { name: 'Not now' }))
  expect(await screen.findByText(/needed before the workspace can open/)).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Agent configuration' })).not.toBeInTheDocument()
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('pull-model'))).toBe(false)
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('update_agents'))).toBe(false)
})

it('streams model pull progress and opens the workspace after readiness is confirmed', async () => {
  let progressController!: ReadableStreamDefaultController<Uint8Array>
  const encoder = new TextEncoder()
  const pullBody = new ReadableStream<Uint8Array>({
    start(controller) {
      progressController = controller
    },
  })
  let statusChecks = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string) => {
      const url = String(input)
      if (url.includes('ollama/dependency-status')) {
        statusChecks += 1
        const installed = statusChecks > 1
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: installed,
          ready: installed,
          error: null,
        }))
      }
      if (url.includes('pull-model')) return new Response(pullBody)
      if (url.includes('get_config'))
        return new Response(JSON.stringify({
          model: 'gemma4:latest',
          model_provider: 'ollama',
          system_prompt_user: 'Extract accurately.',
        }))
      if (url.includes('get_model_providers')) return new Response(JSON.stringify(['ollama']))
      if (url.includes('get_models')) return new Response(JSON.stringify(['gemma4:latest']))
      if (url.includes('update_agents')) return new Response('true')
      return new Response('true')
    }),
  )

  render(<App />)
  fireEvent.click(await screen.findByRole('button', { name: 'Download model' }))
  await act(async () => {
    progressController.enqueue(
      encoder.encode('{"status":"downloading layers","total":100,"completed":40}\n'),
    )
  })
  expect(await screen.findByText('40%')).toBeInTheDocument()
  await act(async () => {
    progressController.enqueue(encoder.encode('{"status":"success"}\n'))
    progressController.close()
  })
  expect(await screen.findByRole('heading', { name: 'Single agent' })).toBeInTheDocument()
  expect(statusChecks).toBe(2)
  expect(vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('update_agents'))).toBe(true)
})

it('shows pull errors and keeps the main workspace blocked', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string) => {
      const url = String(input)
      if (url.includes('ollama/dependency-status'))
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: false,
          ready: false,
          error: null,
        }))
      if (url.includes('pull-model')) return new Response('{"error":"Disk full"}\n')
      return new Response('true')
    }),
  )
  render(<App />)
  fireEvent.click(await screen.findByRole('button', { name: 'Download model' }))
  expect(await screen.findByText('Disk full')).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Agent configuration' })).not.toBeInTheDocument()
})

it('keeps the workspace blocked when Ollama is unavailable and retries readiness', async () => {
  let statusChecks = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string) => {
      const url = String(input)
      if (url.includes('ollama/dependency-status')) {
        statusChecks += 1
        const reachable = statusChecks > 1
        return new Response(JSON.stringify({
          ollama_reachable: reachable,
          required_model: 'gemma4:latest',
          model_installed: reachable,
          ready: reachable,
          error: reachable ? null : 'Cannot retrieve Ollama models. Check that Ollama is running.',
        }))
      }
      return new Response('true')
    }),
  )
  render(<App />)
  expect(await screen.findByText(/Check that Ollama is running/)).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Single agent' })).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Check again' }))
  expect(await screen.findByRole('heading', { name: 'Single agent' })).toBeInTheDocument()
  expect(statusChecks).toBe(2)
})

it('keeps the workspace blocked when a model download is cancelled', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('ollama/dependency-status'))
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: false,
          ready: false,
          error: null,
        }))
      if (url.includes('pull-model'))
        return new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () =>
            reject(new DOMException('Aborted', 'AbortError')),
          )
        })
      return new Response('true')
    }),
  )
  render(<App />)
  fireEvent.click(await screen.findByRole('button', { name: 'Download model' }))
  fireEvent.click(await screen.findByRole('button', { name: 'Cancel download' }))
  expect(await screen.findByText('Model download canceled.')).toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Single agent' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Download required model' })).toBeInTheDocument()
})

it('runs analyser without uploads and keeps its generated table across navigation', async () => {
  await renderReadyWorkspace()
  fireEvent.mouseDown(screen.getByRole('tab', { name: 'Analyser' }), { button: 0, ctrlKey: false })
  fireEvent.click(screen.getByRole('tab', { name: 'Analyser' }))
  await waitFor(() =>
    expect(
      screen.getByText('No dataset uploaded. External-data requests can use the downloader agent.'),
    ).toBeInTheDocument(),
  )
  fireEvent.change(screen.getByLabelText('Analysis prompt'), {
    target: { value: 'Download and analyse data' },
  })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Run analyser' })).toBeEnabled())
  fireEvent.click(screen.getByRole('button', { name: 'Run analyser' }))
  expect(await screen.findByText('123.45')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Download CSV' })).toBeInTheDocument()
  expect(screen.queryByAltText('Generated analysis plot')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('link', { name: 'Run AI Agent OCR' }))
  fireEvent.click(screen.getByRole('link', { name: 'Run single agent' }))
  fireEvent.mouseDown(screen.getByRole('tab', { name: 'Analyser' }), { button: 0, ctrlKey: false })
  fireEvent.click(screen.getByRole('tab', { name: 'Analyser' }))
  expect(await screen.findByText('123.45')).toBeInTheDocument()
})

it('renders independent provider and model forms without manual model entry', async () => {
  window.history.replaceState({}, '', '/')
  await renderReadyWorkspace()
  expect(await screen.findByRole('heading', { name: 'Data extractor' })).toBeInTheDocument()
  await waitFor(() => expect(screen.getAllByLabelText('Model')).toHaveLength(2))
  expect(screen.queryByRole('textbox', { name: 'Model' })).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Temperature')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Top P')).not.toBeInTheDocument()
  await waitFor(() =>
    expect(screen.getAllByRole('button', { name: 'Save configuration' })[1]).toBeEnabled(),
  )
  fireEvent.click(screen.getAllByRole('button', { name: 'Save configuration' })[1])
  await waitFor(() =>
    expect(
      vi.mocked(fetch).mock.calls.some(([url]) => String(url).includes('upload_config')),
    ).toBe(true),
  )
  const openAiSave = vi.mocked(fetch).mock.calls.find(([url]) =>
    String(url).includes('upload_config'),
  )
  expect(JSON.parse(String(openAiSave?.[1]?.body)).model_settings).toEqual({})
  fireEvent.change(screen.getAllByLabelText('Provider')[1], { target: { value: 'ollama' } })
  await waitFor(() => expect(screen.getAllByLabelText('Model')[1]).toHaveValue('gemma4:latest'))
  expect(screen.getAllByLabelText('Temperature')).toHaveLength(1)
  expect(screen.getAllByLabelText('Top P')).toHaveLength(1)
  fireEvent.change(screen.getAllByLabelText('Provider')[1], { target: { value: 'custom' } })
  await waitFor(() => expect(screen.getAllByLabelText('Model')[1]).toHaveValue('custom-model'))
  expect(screen.getAllByLabelText('Model')[0]).toHaveValue('gpt-5')
})

it('edits and saves bounded sampling settings for both Ollama agent forms', async () => {
  const saves: Record<string, unknown>[] = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: string, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('ollama/dependency-status'))
        return new Response(JSON.stringify({
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: true,
          ready: true,
          error: null,
        }))
      if (url.includes('get_config'))
        return new Response(JSON.stringify({
          model: 'gemma4:latest',
          model_provider: 'ollama',
          model_settings: {},
          system_prompt_user: 'Extract accurately.',
        }))
      if (url.includes('get_model_providers'))
        return new Response(JSON.stringify(['openai', 'ollama']))
      if (url.includes('get_models'))
        return new Response(JSON.stringify(url.includes('ollama') ? ['gemma4:latest'] : ['gpt-5']))
      if (url.includes('upload_config')) {
        saves.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
        return new Response('true')
      }
      return new Response('true')
    }),
  )
  window.history.replaceState({}, '', '/')
  await renderReadyWorkspace()

  await waitFor(() => expect(screen.getAllByLabelText('Provider')).toHaveLength(2))
  screen.getAllByLabelText('Provider').forEach((provider) =>
    fireEvent.change(provider, { target: { value: 'ollama' } }),
  )
  await waitFor(() => expect(screen.getAllByLabelText('Temperature')).toHaveLength(2))
  const temperatures = screen.getAllByLabelText('Temperature')
  const topPValues = screen.getAllByLabelText('Top P')
  expect(temperatures[0]).toHaveAttribute('min', '0')
  expect(temperatures[0]).toHaveAttribute('max', '2')
  expect(temperatures[0]).toHaveAttribute('step', '0.1')
  expect(temperatures[0]).toHaveValue('0.01')
  expect(topPValues[0]).toHaveAttribute('min', '0')
  expect(topPValues[0]).toHaveAttribute('max', '1')
  expect(topPValues[0]).toHaveAttribute('step', '0.1')
  expect(topPValues[0]).toHaveValue(1)

  fireEvent.change(temperatures[0], { target: { value: '1.4' } })
  fireEvent.change(topPValues[0], { target: { value: '0.7' } })
  await waitFor(() =>
    expect(screen.getAllByRole('button', { name: 'Save configuration' })[0]).toBeEnabled(),
  )
  fireEvent.click(screen.getAllByRole('button', { name: 'Save configuration' })[0])

  await waitFor(() => expect(saves).toHaveLength(1))
  expect(saves[0].model_settings).toEqual({ temperature: 1.4, top_p: 0.7 })
})

it('uploads documents only after resolution confirmation and supports cancelling and reselection', async () => {
  await renderReadyWorkspace()
  const picker = screen.getByLabelText('Choose documents')
  await waitFor(() => expect(picker).toBeEnabled())
  expect(screen.queryByRole('heading', { name: 'Results' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Upload documents' })).not.toBeInTheDocument()
  const files = [new File(['image'], 'photo.png', { type: 'image/png' })]
  fireEvent.change(picker, { target: { files } })
  expect(await screen.findByRole('dialog', { name: 'Image resolution' })).toBeVisible()
  expect(
    vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('uploadimages')),
  ).toHaveLength(0)
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(picker).toBeEnabled())
  fireEvent.change(picker, { target: { files } })
  fireEvent.click(screen.getByRole('radio', { name: 'Small' }))
  fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
  await screen.findByText('photo.png')
  const call = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes('uploadimages'))
  const body = call?.[1]?.body
  expect(body).toBeInstanceOf(FormData)
  expect((body as FormData).get('resize')).toBe('small')
  expect(
    vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('uploadimages')),
  ).toHaveLength(1)
})

it('automatically uploads a selected dataset without a resolution dialog', async () => {
  await renderReadyWorkspace()
  fireEvent.mouseDown(screen.getByRole('tab', { name: 'Analyser' }), { button: 0 })
  const picker = await screen.findByLabelText('Choose a dataset')
  await waitFor(() => expect(picker).toBeEnabled())
  fireEvent.change(picker, { target: { files: [new File(['value\n1'], 'data.csv')] } })
  await screen.findByText('data.csv')
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(
    vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('uploaddatafile')),
  ).toHaveLength(1)
})

it('keeps running feedback until results finish downloading and scrolls only once', async () => {
  const original = fetch
  let completeRun!: (response: Response) => void
  let completeDownload!: (response: Response) => void
  const generation = new Promise<Response>((resolve) => {
    completeRun = resolve
  })
  const download = new Promise<Response>((resolve) => {
    completeDownload = resolve
  })
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string, init?: RequestInit) => {
      if (url.includes('run_single_agent')) return generation
      if (url.includes('download_data')) return download
      return original(url, init)
    }),
  )
  await renderReadyWorkspace()
  fireEvent.mouseDown(screen.getByRole('tab', { name: 'Analyser' }), { button: 0 })
  fireEvent.change(await screen.findByLabelText('Analysis prompt'), {
    target: { value: 'Calculate totals' },
  })
  const run = screen.getByRole('button', { name: 'Run analyser' })
  await waitFor(() => expect(run).toBeEnabled())
  fireEvent.click(run)
  expect(screen.getByRole('button', { name: 'Running' })).toBeDisabled()
  expect(screen.getAllByText('Generating analysis / plot').length).toBeGreaterThan(0)
  expect(screen.queryByRole('heading', { name: 'Results' })).not.toBeInTheDocument()
  await act(async () => completeRun(new Response(JSON.stringify({ df_file_path: 'result.csv' }))))
  expect(screen.getAllByText('Loading results').length).toBeGreaterThan(0)
  await act(async () => completeDownload(new Response('Total\n42\n')))
  expect(await screen.findByText('42')).toBeInTheDocument()
  await waitFor(() => expect(Element.prototype.scrollIntoView).toHaveBeenCalledTimes(1))
  fireEvent.change(screen.getByLabelText('Filter result rows'), { target: { value: '42' } })
  expect(Element.prototype.scrollIntoView).toHaveBeenCalledTimes(1)
  expect(screen.queryByText('Please wait while your request is processed.')).not.toBeInTheDocument()
})

it('releases a pending document choice on route change without sending an upload', async () => {
  await renderReadyWorkspace()
  const picker = screen.getByLabelText('Choose documents')
  await waitFor(() => expect(picker).toBeEnabled())
  fireEvent.change(picker, { target: { files: [new File(['pdf'], 'document.pdf')] } })
  await screen.findByRole('dialog')
  fireEvent.click(screen.getByRole('link', { name: 'Run AI Agent OCR' }))
  await waitFor(() => expect(screen.getByLabelText('Choose documents')).toBeEnabled())
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(
    vi.mocked(fetch).mock.calls.filter(([url]) => String(url).includes('uploadpdf')),
  ).toHaveLength(0)
})

it('validates dropped batches and uploads dropped PDFs after confirmation', async () => {
  await renderReadyWorkspace()
  const picker = screen.getByLabelText('Choose documents')
  await waitFor(() => expect(picker).toBeEnabled())
  const dropzone = picker.closest('.dropzone')!
  const pdf = new File(['pdf'], 'statement.pdf')
  fireEvent.drop(dropzone, { dataTransfer: { files: [pdf, new File([], 'photo.png')] } })
  expect(screen.getByText('Select one PDF without additional images.')).toBeInTheDocument()
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  fireEvent.drop(dropzone, { dataTransfer: { files: [pdf] } })
  fireEvent.click(await screen.findByRole('button', { name: 'Continue' }))
  await screen.findByText('statement.pdf')
  const upload = vi.mocked(fetch).mock.calls.find(([url]) => String(url).includes('uploadpdf'))
  const body = upload?.[1]?.body
  expect(body).toBeInstanceOf(FormData)
  expect((body as FormData).get('resize')).toBe('medium')
})

it('stops model feedback on failure and does not reveal or scroll to results', async () => {
  const original = fetch
  vi.stubGlobal(
    'fetch',
    vi.fn((url: string, init?: RequestInit) =>
      url.includes('run_single_agent')
        ? Promise.resolve(new Response(JSON.stringify({ error_message: 'Model unavailable' })))
        : original(url, init),
    ),
  )
  await renderReadyWorkspace()
  fireEvent.mouseDown(screen.getByRole('tab', { name: 'Analyser' }), { button: 0 })
  fireEvent.change(await screen.findByLabelText('Analysis prompt'), {
    target: { value: 'Analyse' },
  })
  const run = screen.getByRole('button', { name: 'Run analyser' })
  await waitFor(() => expect(run).toBeEnabled())
  fireEvent.click(run)
  expect(await screen.findByText('Model unavailable')).toBeInTheDocument()
  expect(screen.queryByText('Please wait while your request is processed.')).not.toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Results' })).not.toBeInTheDocument()
  expect(Element.prototype.scrollIntoView).not.toHaveBeenCalled()
  expect(run).toBeEnabled()
})
