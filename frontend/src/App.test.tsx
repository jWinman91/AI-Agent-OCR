import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import App from './App'

beforeEach(() => {
  vi.mocked(Element.prototype.scrollIntoView).mockClear()
  window.history.replaceState({}, '', '/single')
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string) => {
      if (url.includes('get_config'))
        return new Response(
          JSON.stringify({
            model: 'gpt-5',
            model_provider: 'openai',
            system_prompt_user: 'Extract accurately.',
          }),
        )
      if (url.includes('tags'))
        return new Response(JSON.stringify({ models: [{ name: 'gemma4:latest' }] }))
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

it('runs analyser without uploads and keeps its generated table across navigation', async () => {
  render(<App />)
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
  render(<App />)
  expect(await screen.findByRole('heading', { name: 'Data extractor' })).toBeInTheDocument()
  await waitFor(() => expect(screen.getAllByLabelText('Model')).toHaveLength(2))
  expect(screen.queryByRole('textbox', { name: 'Model' })).not.toBeInTheDocument()
  fireEvent.change(screen.getAllByLabelText('Provider')[1], { target: { value: 'ollama' } })
  await waitFor(() => expect(screen.getAllByLabelText('Model')[1]).toHaveValue('gemma4:latest'))
})

it('uploads documents only after resolution confirmation and supports cancelling and reselection', async () => {
  render(<App />)
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
  render(<App />)
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
  render(<App />)
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
  render(<App />)
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
  render(<App />)
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
  render(<App />)
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
