import '@fontsource-variable/public-sans'
import * as Tabs from '@radix-ui/react-tabs'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import type { ColumnDef } from '@tanstack/react-table'
import {
    flexRender,
    getCoreRowModel,
    getPaginationRowModel,
    getSortedRowModel,
    useReactTable,
} from '@tanstack/react-table'
import {
    ArrowDownToLine,
    ArrowLeft,
    ArrowRight,
    Check,
    FileImage,
    FileText,
    LoaderCircle,
    Play,
    RefreshCw,
    ScanLine,
    Settings2,
    SlidersHorizontal,
    X,
} from 'lucide-react'
import type { ReactNode } from 'react'
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import type { AgentConfig, AgentName, ResultView, Workflow } from './api'
import { AgentFailure, api, loadResult, message } from './api'
import './App.css'

type RunState = {
    prompt: string
    uploaded: string[]
    error?: string
    previousError?: string
    result?: ResultView
    resultRevision?: number
}
type Workspace = {
    busy: string
    busySince: number
    activeWorkflow?: Workflow
    selectedAgent: Workflow
    selectAgent: (workflow: Workflow) => void
    initializeError: string
    ready: boolean
    states: Record<Workflow, RunState>
    update: (workflow: Workflow, patch: Partial<RunState>) => void
    operation: (label: string, work: () => Promise<void>, owner?: symbol) => Promise<boolean>
    reserve: () => symbol | undefined
    release: (owner: symbol) => void
    activity: (label: string, workflow?: Workflow) => void
    present: (workflow: Workflow, result: ResultView) => boolean
    initialize: () => Promise<void>
    consume: (workflow: Workflow) => void
}
const Context = createContext<Workspace | null>(null)
function useWorkspace() {
    const value = useContext(Context)
    if (!value) throw new Error('Missing workspace provider')
    return value
}
const queryClient = new QueryClient({
    defaultOptions: {
        queries: { retry: false, refetchOnWindowFocus: false, staleTime: 30000 },
        mutations: { retry: false },
    },
})

function WorkspaceProvider({ children }: { children: ReactNode }) {
    const [busy, setBusy] = useState('')
    const lock = useRef<symbol | null>(null)
    const presented = useRef<Partial<Record<Workflow, ResultView>>>({})
    const [busySince, setBusySince] = useState(0)
    const [activeWorkflow, setActiveWorkflow] = useState<Workflow>()
    const [selectedAgent, selectAgent] = useState<Workflow>('extractor')
    const startup = useRef<Promise<void> | null>(null)
    const [ready, setReady] = useState(false)
    const [initializeError, setInitializeError] = useState('')
    const [states, setStates] = useState<Record<Workflow, RunState>>({
        pipeline: { prompt: '', uploaded: [] },
        extractor: { prompt: '', uploaded: [] },
        analyser: { prompt: '', uploaded: [] },
    })
    const update = (workflow: Workflow, patch: Partial<RunState>) =>
        setStates((current) => ({
            ...current,
            [workflow]: {
                ...current[workflow],
                ...patch,
                ...(patch.result ? { resultRevision: (current[workflow].resultRevision || 0) + 1 } : {}),
            },
        }))
    const consume = (workflow: Workflow) =>
        setStates((current) => {
            const next = { ...current }
            const workflows: Workflow[] =
                workflow === 'analyser' ? ['analyser'] : ['pipeline', 'extractor']
            workflows.forEach((item) => {
                next[item] = { ...next[item], uploaded: [] }
            })
            return next
        })
    const release = useCallback((owner: symbol) => {
        if (lock.current !== owner) return
        lock.current = null
        setBusy('')
        setActiveWorkflow(undefined)
    }, [])
    const reserve = useCallback(() => {
        if (lock.current) return undefined
        const owner = Symbol('resolution')
        lock.current = owner
        setBusy('Choosing image resolution')
        return owner
    }, [])
    const activity = (label: string, workflow?: Workflow) => {
        setBusy(label)
        setActiveWorkflow(workflow)
    }
    const present = useCallback((workflow: Workflow, result: ResultView) => {
        if (presented.current[workflow] === result) return false
        presented.current[workflow] = result
        return true
    }, [])
    const operation = async (label: string, work: () => Promise<void>, owner?: symbol) => {
        if (lock.current && lock.current !== owner) return false
        const token = owner || Symbol('operation')
        lock.current = token
        setBusy(label)
        setBusySince(Date.now())
        try {
            await work()
        } finally {
            release(token)
        }
        return true
    }
    const initialize = async () => {
        try {
            await api.initialize()
            setReady(true)
            setInitializeError('')
        } catch (error) {
            setReady(false)
            setInitializeError(message(error))
        }
    }
    useEffect(() => {
        startup.current ??= api.initialize()
        void startup.current
            .then(() => setReady(true))
            .catch((error) => setInitializeError(message(error)))
    }, [])
    return (
        <Context.Provider
            value={{
                busy,
                busySince,
                activeWorkflow,
                selectedAgent,
                selectAgent,
                ready,
                initializeError,
                states,
                update,
                operation,
                reserve,
                release,
                activity,
                present,
                initialize,
                consume,
            }}
        >
            {children}
        </Context.Provider>
    )
}
function Notice({
    children,
    tone = 'error',
}: {
    children: ReactNode
    tone?: 'error' | 'warning' | 'success'
}) {
    return (
        <div className={`notice ${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
            {children}
        </div>
    )
}
function Download({ blob }: { blob: Blob }) {
    return (
        <button
            className="button secondary"
            onClick={() => {
                const url = URL.createObjectURL(blob)
                const anchor = document.createElement('a')
                anchor.href = url
                anchor.download = 'results.csv'
                anchor.click()
                setTimeout(() => URL.revokeObjectURL(url), 1000)
            }}
        >
            <ArrowDownToLine size={16} />
            Download CSV
        </button>
    )
}
function Plot({ blob }: { blob: Blob }) {
    const image = useRef<HTMLImageElement>(null)
    useEffect(() => {
        const url = URL.createObjectURL(blob)
        if (image.current) image.current.src = url
        return () => URL.revokeObjectURL(url)
    }, [blob])
    return <img ref={image} className="result-plot" alt="Generated analysis plot" />
}
function Results({
    result,
    preview,
    sideBySide,
}: {
    result: ResultView
    preview: boolean
    sideBySide: boolean
}) {
    const [filter, setFilter] = useState('')
    const columns: ColumnDef<string[]>[] = (result?.columns || []).map((title, index) => ({
        id: String(index),
        header: title,
        accessorFn: (row) => row[index] || '',
    }))
    const rows = result?.rows || []
    const filtered = rows.filter((row) =>
        row.some((cell) => cell.toLowerCase().includes(filter.toLowerCase())),
    )
    const table = useReactTable({
        autoResetPageIndex: false,
        data: preview ? filtered.slice(0, 5) : filtered,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        getPaginationRowModel: getPaginationRowModel(),
        initialState: { pagination: { pageSize: 20 } },
    })
    return (
        <section className="results">
            <div className="section-heading">
                <h2>Results</h2>
                {result.csv !== undefined && (
                    <Download blob={new Blob([result.csv], { type: 'text/csv;charset=utf-8' })} />
                )}
            </div>
            {result.errors.map((error) => (
                <Notice key={error}>{error}</Notice>
            ))}
            <div
                className={`result-content ${sideBySide && result.csv !== undefined && result.plot ? 'split-results' : ''}`}
            >
                {result.csv !== undefined && (
                    <div className="result-data">
                        <div className="table-tools">
                            <span>
                                {rows.length} rows{preview ? ' · Preview' : ''}
                            </span>
                            <input
                                aria-label="Filter result rows"
                                placeholder="Filter rows"
                                value={filter}
                                onChange={(event) => {
                                    setFilter(event.target.value)
                                    table.setPageIndex(0)
                                }}
                            />
                        </div>
                        <div className="table-scroll">
                            <table>
                                <thead>
                                    {table.getHeaderGroups().map((group) => (
                                        <tr key={group.id}>
                                            {group.headers.map((header) => (
                                                <th key={header.id}>
                                                    <button onClick={header.column.getToggleSortingHandler()}>
                                                        {flexRender(header.column.columnDef.header, header.getContext())}
                                                        {header.column.getIsSorted() === 'asc'
                                                            ? ' ↑'
                                                            : header.column.getIsSorted() === 'desc'
                                                                ? ' ↓'
                                                                : ''}
                                                    </button>
                                                </th>
                                            ))}
                                        </tr>
                                    ))}
                                </thead>
                                <tbody>
                                    {table.getRowModel().rows.map((row) => (
                                        <tr key={row.id}>
                                            {row.getVisibleCells().map((cell) => (
                                                <td key={cell.id}>
                                                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                        {!preview && (
                            <div className="pagination">
                                <span>
                                    Page {table.getState().pagination.pageIndex + 1} of{' '}
                                    {Math.max(1, table.getPageCount())}
                                </span>
                                <button
                                    title="Previous page"
                                    aria-label="Previous page"
                                    disabled={!table.getCanPreviousPage()}
                                    onClick={() => table.previousPage()}
                                >
                                    <ArrowLeft size={16} />
                                </button>
                                <button
                                    title="Next page"
                                    aria-label="Next page"
                                    disabled={!table.getCanNextPage()}
                                    onClick={() => table.nextPage()}
                                >
                                    <ArrowRight size={16} />
                                </button>
                            </div>
                        )}
                    </div>
                )}
                {result.plot && (
                    <div className="result-image">
                        <Plot blob={result.plot} />
                    </div>
                )}
            </div>
            {result.summary && <p className="summary">{result.summary}</p>}
        </section>
    )
}

function RunningStatus({ label, since }: { label: string; since: number }) {
    const [now, setNow] = useState(Date.now)
    useEffect(() => {
        const timer = window.setInterval(() => setNow(Date.now()), 1000)
        return () => window.clearInterval(timer)
    }, [])
    const elapsed = Math.max(0, Math.floor((now - since) / 1000))
    return (
        <div className="running-status" role="status">
            <div className="running-symbol">
                <LoaderCircle className="spin" size={28} />
            </div>
            <div>
                <strong>{label}</strong>
                <span>Please wait while your request is processed.</span>
            </div>
            <time aria-hidden="true">
                {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, '0')}
            </time>
            <div className="activity-track" aria-hidden="true">
                <span />
            </div>
        </div>
    )
}

function ResolutionDialog({
    files,
    confirm,
    cancel,
}: {
    files: File[]
    confirm: (resolution: string) => void
    cancel: () => void
}) {
    const dialog = useRef<HTMLDialogElement>(null)
    const [resolution, setResolution] = useState('medium')
    useEffect(() => {
        const node = dialog.current
        node?.showModal()
        return () => node?.close()
    }, [])
    return (
        <dialog
            ref={dialog}
            className="resolution-dialog"
            aria-labelledby="resolution-title"
            onCancel={(event) => {
                event.preventDefault()
                cancel()
            }}
        >
            <div className="section-heading">
                <h2 id="resolution-title">Image resolution</h2>
                <button className="icon-button" aria-label="Close resolution dialog" onClick={cancel}>
                    <X size={18} />
                </button>
            </div>
            <p className="dialog-files">{files.map((file) => file.name).join(', ')}</p>
            <fieldset>
                <legend>Resolution</legend>
                {['original', 'medium', 'small'].map((value) => (
                    <label
                        className={`resolution-option ${resolution === value ? 'selected' : ''}`}
                        key={value}
                    >
                        <input
                            type="radio"
                            name="resolution"
                            value={value}
                            autoFocus={value === 'medium'}
                            checked={resolution === value}
                            onChange={() => setResolution(value)}
                        />
                        <span>{value[0].toUpperCase() + value.slice(1)}</span>
                    </label>
                ))}
            </fieldset>
            <div className="form-actions">
                <button className="button secondary" onClick={cancel}>
                    Cancel
                </button>
                <button className="button primary" onClick={() => confirm(resolution)}>
                    Continue
                </button>
            </div>
        </dialog>
    )
}

function WorkflowView({ workflow }: { workflow: Workflow }) {
    const {
        states,
        update,
        busy,
        busySince,
        activeWorkflow,
        ready,
        operation,
        consume,
        reserve,
        release,
        activity,
        present,
    } = useWorkspace()
    const state = states[workflow]
    const [pending, setPending] = useState<File[]>([])
    const pendingOwner = useRef<symbol | undefined>(undefined)
    const [selectionError, setSelectionError] = useState('')
    const input = useRef<HTMLInputElement>(null)
    const resultSection = useRef<HTMLDivElement>(null)
    const analyser = workflow === 'analyser'
    const disabled = Boolean(busy) || !ready
    const running = activeWorkflow === workflow
    useEffect(
        () => () => {
            if (pendingOwner.current) release(pendingOwner.current)
        },
        [release],
    )
    useEffect(() => {
        if (!state.result) return
        const result = state.result
        const frame = requestAnimationFrame(() => {
            const node = resultSection.current
            if (node?.isConnected && present(workflow, result)) {
                node.scrollIntoView({
                    behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
                        ? 'instant'
                        : 'smooth',
                    block: 'start',
                })
            }
        })
        return () => cancelAnimationFrame(frame)
    }, [state.result, workflow, present])
    const upload = (files: File[], resolution: string, owner?: symbol) =>
        operation(
            'Uploading files',
            async () => {
                update(workflow, { error: undefined })
                try {
                    await api.upload(workflow, files, resolution)
                    consume(workflow)
                    update(workflow, { uploaded: files.map((file) => file.name) })
                } catch (error) {
                    consume(workflow)
                    update(workflow, { error: message(error) })
                }
            },
            owner,
        )
    const selectFiles = (files: File[]) => {
        if (input.current) input.current.value = ''
        if (!files.length || disabled) return
        setSelectionError('')
        const pdf = files.some((file) => /\.pdf$/i.test(file.name))
        if ((analyser && files.length !== 1) || (!analyser && pdf && files.length !== 1)) {
            setSelectionError(
                analyser ? 'Select one dataset at a time.' : 'Select one PDF without additional images.',
            )
            return
        }
        if (
            files.some((file) => !(analyser ? /\.(csv|xlsx)$/i : /\.(pdf|png|jpe?g)$/i).test(file.name))
        ) {
            setSelectionError('Unsupported file type.')
            return
        }
        if (analyser) {
            void upload(files, 'medium')
            return
        }
        const owner = reserve()
        if (!owner) {
            setSelectionError('Another operation is in progress. Try again when it finishes.')
            return
        }
        pendingOwner.current = owner
        setPending(files)
    }
    const cancel = () => {
        if (pendingOwner.current) release(pendingOwner.current)
        pendingOwner.current = undefined
        setPending([])
        requestAnimationFrame(() => input.current?.focus())
    }
    const confirm = (resolution: string) => {
        const owner = pendingOwner.current
        if (!owner) return
        const files = pending
        pendingOwner.current = undefined
        setPending([])
        void upload(files, resolution, owner)
    }
    const run = () =>
        operation('Running agents', async () => {
            activity(
                workflow === 'extractor'
                    ? 'Extracting data'
                    : analyser
                        ? 'Generating analysis / plot'
                        : 'Running OCR pipeline',
                workflow,
            )
            update(workflow, { error: undefined })
            try {
                const result = await api.run(workflow, state.prompt, state.previousError)
                update(workflow, { previousError: undefined })
                activity('Loading results', workflow)
                update(workflow, { result: await loadResult(result, workflow) })
            } catch (error) {
                update(workflow, {
                    error: message(error),
                    ...(error instanceof AgentFailure ? { previousError: message(error) } : {}),
                })
            } finally {
                consume(workflow)
            }
        })
    return (
        <div className="run-workspace" aria-busy={running}>
            <div className="workflow-grid">
                <section className="document-pane">
                    <div className="section-heading">
                        <h2>{analyser ? 'Dataset' : 'Documents'}</h2>
                        <span className="step-label">01</span>
                    </div>
                    <div
                        className="dropzone"
                        onDragOver={(event) => event.preventDefault()}
                        onDrop={(event) => {
                            event.preventDefault()
                            selectFiles(Array.from(event.dataTransfer.files))
                        }}
                    >
                        {analyser ? <FileText size={34} /> : <FileImage size={34} />}
                        <label className="file-label">
                            {analyser ? 'Choose a dataset' : 'Choose documents'}
                            <input
                                ref={input}
                                disabled={disabled}
                                type="file"
                                accept={analyser ? '.csv,.xlsx' : '.pdf,.png,.jpg,.jpeg'}
                                multiple={!analyser}
                                onChange={(event) => selectFiles(Array.from(event.target.files || []))}
                            />
                        </label>
                        <span>{analyser ? 'CSV or Excel' : 'PDF, PNG or JPEG'}</span>
                    </div>
                    {selectionError && <Notice>{selectionError}</Notice>}
                    {busy === 'Uploading files' && (
                        <div className="upload-status" role="status">
                            <LoaderCircle className="spin" size={18} />
                            Uploading files
                        </div>
                    )}
                    {state.uploaded.length > 0 && (
                        <ul className="uploaded-files">
                            {state.uploaded.map((filename) => (
                                <li key={filename}>
                                    <FileText size={18} />
                                    <span>{filename}</span>
                                    <Check size={16} />
                                </li>
                            ))}
                        </ul>
                    )}
                    {analyser && !state.uploaded.length && (
                        <Notice tone="warning">
                            No dataset uploaded. External-data requests can use the downloader agent.
                        </Notice>
                    )}
                </section>
                <section className="prompt-pane">
                    <div className="section-heading">
                        <h2>Your request</h2>
                        <span className="step-label">02</span>
                    </div>
                    <label className="field prompt-field">
                        {workflow === 'extractor' ? 'Extraction prompt' : 'Analysis prompt'}
                        <textarea
                            value={state.prompt}
                            disabled={disabled}
                            rows={10}
                            placeholder={
                                workflow === 'extractor'
                                    ? 'What would you like to extract?'
                                    : 'What would you like to analyse?'
                            }
                            onChange={(event) => update(workflow, { prompt: event.target.value })}
                        />
                    </label>
                    {state.error && (
                        <Notice>
                            <span className="error-text">{state.error}</span>
                            <button
                                className="icon-button"
                                title="Dismiss error"
                                aria-label="Dismiss error"
                                onClick={() => update(workflow, { error: undefined })}
                            >
                                <X size={16} />
                            </button>
                        </Notice>
                    )}
                    <div className="run-actions">
                        <button
                            className="button primary run-button"
                            disabled={disabled || !state.prompt.trim() || (!analyser && !state.uploaded.length)}
                            onClick={() => void run()}
                        >
                            {running ? <LoaderCircle className="spin" size={18} /> : <Play size={18} />}
                            {running
                                ? 'Running'
                                : `Run ${workflow === 'pipeline' ? 'pipeline' : workflow === 'extractor' ? 'extractor' : 'analyser'}`}
                        </button>
                    </div>
                </section>
            </div>
            {running && <RunningStatus label={busy} since={busySince} />}
            {state.result && (
                <div ref={resultSection} className="results-section">
                    <Results
                        key={`${workflow}-${state.resultRevision}`}
                        result={state.result}
                        preview={workflow === 'extractor'}
                        sideBySide={analyser}
                    />
                </div>
            )}
            {pending.length > 0 && <ResolutionDialog files={pending} confirm={confirm} cancel={cancel} />}
        </div>
    )
}

function AgentForm({ agent }: { agent: AgentName }) {
    const { busy, ready, operation } = useWorkspace()
    const config = useQuery({ queryKey: ['config', agent], queryFn: () => api.config(agent) })
    const providers = useQuery({ queryKey: ['model-providers'], queryFn: api.providers })
    const [edits, setDraft] = useState<AgentConfig>()
    const draft = edits || config.data
    const models = useQuery({
        queryKey: ['models', draft?.model_provider],
        queryFn: () => api.models(draft?.model_provider || ''),
        enabled: Boolean(draft),
    })
    const [key, setKey] = useState('')
    const [notice, setNotice] = useState('')
    const [error, setError] = useState('')
    if (config.isPending || providers.isPending)
        return (
            <section className="config-section">
                <LoaderCircle className="spin" size={18} /> Loading configuration
            </section>
        )
    if (config.error || providers.error || !draft)
        return (
            <section className="config-section">
                <Notice>{message(config.error || providers.error)}</Notice>
                <button className="button secondary" onClick={() => {
                    void config.refetch()
                    void providers.refetch()
                }}>
                    <RefreshCw size={16} />
                    Retry
                </button>
            </section>
        )
    const available = models.error ? [] : models.data || []
    const selectedModel = draft.model || available[0] || ''
    const missing = !available.includes(selectedModel)
    const providerMissing = !providers.data?.includes(draft.model_provider)
    const samplingValue = (name: 'temperature' | 'top_p', fallback: number, max: number) => {
        const value = draft.model_settings?.[name]
        return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= max
            ? value
            : fallback
    }
    const temperature = samplingValue('temperature', 0.01, 2)
    const topP = samplingValue('top_p', 1, 1)
    const change = (patch: Partial<AgentConfig>) => {
        setDraft({ ...draft, ...patch })
        setNotice('')
        setError('')
    }
    const changeSamplingValue = (name: 'temperature' | 'top_p', value: number, max: number) => {
        if (!Number.isFinite(value) || value < 0 || value > max) return
        change({ model_settings: { ...draft.model_settings, [name]: value } })
    }
    const save = () =>
        operation('Saving configuration', async () => {
            setError('')
            setNotice('')
            try {
                if (draft.model_provider === 'openai' && key) await api.setKey(key)
                await api.save(agent, {
                    ...draft,
                    model: selectedModel,
                    model_settings:
                        draft.model_provider === 'openai'
                            ? {}
                            : draft.model_provider === 'ollama'
                                ? {
                                    ...draft.model_settings,
                                    temperature,
                                    top_p: topP,
                                }
                                : draft.model_settings,
                })
                await config.refetch()
                setNotice('Configuration saved.')
            } catch (failure) {
                setError(message(failure))
            }
        })
    const reset = () => {
        if (
            !window.confirm(
                `Reset ${agent === 'analyser' ? 'analyser' : 'data extractor'} configuration to its defaults?`,
            )
        )
            return
        void operation('Resetting configuration', async () => {
            setError('')
            setNotice('')
            try {
                await api.reset(agent)
                const fresh = await config.refetch()
                if (fresh.error) throw fresh.error
                if (fresh.data) setDraft(fresh.data)
                setNotice('Configuration reset.')
            } catch (failure) {
                setError(message(failure))
            }
        })
    }
    const disabled = Boolean(busy) || !ready
    return (
        <section className="config-section">
            <div className="section-heading">
                <h2>{agent === 'analyser' ? 'Analyser' : 'Data extractor'}</h2>
                <span className="agent-tag">
                    {agent === 'analyser' ? 'Analysis & plotting' : 'Vision & OCR'}
                </span>
            </div>
            <div className="config-fields">
                <label className="field">
                    Provider
                    <select
                        value={draft.model_provider}
                        disabled={disabled}
                        onChange={(event) => {
                            change({
                                model_provider: event.target.value,
                                model: '',
                            })
                        }}
                    >
                        {providerMissing && <option value={draft.model_provider} disabled>
                            {draft.model_provider} (unavailable)
                        </option>}
                        {providers.data?.map((provider) => (
                            <option key={provider} value={provider}>
                                {provider === 'openai' ? 'OpenAI' : provider === 'ollama' ? 'Ollama' : provider}
                            </option>
                        ))}
                    </select>
                </label>
                <label className="field">
                    Model
                    <select
                        value={missing ? '' : selectedModel}
                        disabled={disabled || !available.length}
                        onChange={(event) => change({ model: event.target.value })}
                    >
                        {missing && (
                            <option value="" disabled>
                                Select a model
                            </option>
                        )}
                        {available.map((model) => (
                            <option key={model}>{model}</option>
                        ))}
                    </select>
                </label>
            </div>
            {draft.model_provider === 'ollama' && (
                <div className="sampling-settings">
                    <label className="field sampling-field">
                        <span className="sampling-label">
                            Temperature <span className="sampling-value" aria-live="polite">{temperature}</span>
                        </span>
                        <input
                            id={`temperature-${agent}`}
                            aria-label="Temperature"
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={temperature}
                            disabled={disabled}
                            onChange={(event) =>
                                changeSamplingValue('temperature', Number(event.target.value), 2)
                            }
                        />
                    </label>
                    <label className="field sampling-field">
                        Top P
                        <input
                            aria-label="Top P"
                            type="number"
                            min="0"
                            max="1"
                            step="0.1"
                            value={topP}
                            disabled={disabled}
                            onChange={(event) =>
                                changeSamplingValue('top_p', event.target.valueAsNumber, 1)
                            }
                        />
                    </label>
                </div>
            )}
            <div className="model-status">
                {models.error ? (
                    <Notice>{message(models.error)}</Notice>
                ) : models.isPending ? (
                    <span><LoaderCircle className="spin" size={16} /> Loading models</span>
                ) : !models.isPending && !available.length ? (
                    <Notice tone="warning">No models available for {draft.model_provider}.</Notice>
                ) : (
                    missing && (
                        <Notice tone="warning">
                            The configured model is unavailable. Select a replacement.
                        </Notice>
                    )
                )}
                <button
                    className="icon-button"
                    title="Refresh models"
                    aria-label="Refresh models"
                    disabled={models.isFetching || disabled}
                    onClick={() => void models.refetch()}
                >
                    <RefreshCw size={16} className={models.isFetching ? 'spin' : ''} />
                </button>
            </div>
            {draft.model_provider === 'openai' && (
                <label className="field">
                    OpenAI API key
                    <input
                        type="password"
                        autoComplete="off"
                        value={key}
                        disabled={disabled}
                        onChange={(event) => setKey(event.target.value)}
                    />
                </label>
            )}
            <label className="field">
                System prompt
                <textarea
                    rows={13}
                    value={draft.system_prompt_user}
                    disabled={disabled}
                    onChange={(event) => change({ system_prompt_user: event.target.value })}
                />
            </label>
            {error && <Notice>{error}</Notice>}
            {notice && (
                <Notice tone="success">
                    <Check size={16} />
                    {notice}
                </Notice>
            )}
            <div className="form-actions">
                <button
                    className="button primary"
                    disabled={disabled || missing || providerMissing}
                    onClick={() => void save()}
                >
                    <Check size={16} />
                    Save configuration
                </button>
                <button className="button subtle danger" disabled={disabled} onClick={reset}>
                    <RefreshCw size={16} />
                    Reset
                </button>
            </div>
        </section>
    )
}
function PageHeading({ title, subtitle }: { title: string; subtitle: string }) {
    return (
        <header className="page-heading">
            <span className="eyebrow">WORKSPACE</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </header>
    )
}
function Configuration() {
    return (
        <>
            <PageHeading title="Agent configuration" subtitle="Models and system prompts" />
            <div className="configuration-grid">
                <AgentForm agent="data_extractor" />
                <AgentForm agent="analyser" />
            </div>
        </>
    )
}
function SingleAgent() {
    const { selectedAgent, selectAgent, busy } = useWorkspace()
    return (
        <>
            <Tabs.Root value={selectedAgent} onValueChange={(value) => selectAgent(value as Workflow)}>
                <Tabs.List className="tabs" aria-label="Agent mode">
                    <Tabs.Trigger value="extractor" disabled={busy === 'Choosing image resolution'}>
                        <ScanLine size={16} />
                        Data extractor
                    </Tabs.Trigger>
                    <Tabs.Trigger value="analyser" disabled={busy === 'Choosing image resolution'}>
                        <SlidersHorizontal size={16} />
                        Analyser
                    </Tabs.Trigger>
                </Tabs.List>
                <PageHeading
                    title="Single agent"
                    subtitle={
                        selectedAgent === 'extractor'
                            ? 'Extract structured data from your documents'
                            : 'Turn your data into insights'
                    }
                />
                <Tabs.Content value="extractor">
                    <WorkflowView workflow="extractor" />
                </Tabs.Content>
                <Tabs.Content value="analyser">
                    <WorkflowView workflow="analyser" />
                </Tabs.Content>
            </Tabs.Root>
        </>
    )
}
function Shell() {
    const { busy, ready, initializeError, initialize, operation } = useWorkspace()
    return (
        <div className="app-shell">
            <aside className="sidebar">
                <NavLink to="/" className="brand">
                    <div className="brand-icon">
                        <ScanLine size={23} />
                    </div>
                    <span>
                        AI Agent OCR<small>Document workspace</small>
                    </span>
                </NavLink>
                <span className="nav-label">WORKSPACE</span>
                <nav>
                    <NavLink to="/" end>
                        <Settings2 size={18} />
                        Configure agents
                    </NavLink>
                    <NavLink to="/pipeline">
                        <FileImage size={18} />
                        Run AI Agent OCR
                    </NavLink>
                    <NavLink to="/single">
                        <SlidersHorizontal size={18} />
                        Run single agent
                    </NavLink>
                </nav>
                <div className="sidebar-footer">
                    {busy ? (
                        <>
                            <LoaderCircle className="spin" size={15} />
                            <span role="status">{busy}</span>
                        </>
                    ) : (
                        <>
                            <span className={`status-dot ${ready ? 'online' : ''}`} />
                            {ready ? 'Backend connected' : 'Backend unavailable'}
                        </>
                    )}
                </div>
            </aside>
            <div className="main-area">
                <main>
                    {initializeError && (
                        <Notice>
                            {initializeError}
                            <button
                                className="button secondary"
                                disabled={Boolean(busy)}
                                onClick={() => void operation('Connecting to backend', initialize)}
                            >
                                <RefreshCw size={15} />
                                Reconnect
                            </button>
                        </Notice>
                    )}
                    <Routes>
                        <Route path="/" element={<Configuration />} />
                        <Route
                            path="/pipeline"
                            element={
                                <>
                                    <PageHeading
                                        title="Run AI Agent OCR"
                                        subtitle="Document extraction, analysis and plotting"
                                    />
                                    <WorkflowView workflow="pipeline" />
                                </>
                            }
                        />
                        <Route path="/single" element={<SingleAgent />} />
                        <Route
                            path="*"
                            element={
                                <>
                                    <h1>Page not found</h1>
                                    <NavLink to="/">Return to workspace</NavLink>
                                </>
                            }
                        />
                    </Routes>
                </main>
            </div>
        </div>
    )
}
export default function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <BrowserRouter>
                <WorkspaceProvider>
                    <Shell />
                </WorkspaceProvider>
            </BrowserRouter>
        </QueryClientProvider>
    )
}
