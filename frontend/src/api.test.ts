import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, loadResult, outputPath } from './api'

afterEach(() => vi.unstubAllGlobals())
describe('backend contracts', () => {
    it('validates Ollama dependency status', async () => {
        const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
            ollama_reachable: true,
            required_model: 'gemma4:latest',
            model_installed: true,
            ready: true,
            error: null,
        })))
        vi.stubGlobal('fetch', fetch)
        expect(await api.ollamaDependencyStatus()).toEqual({
            ollama_reachable: true,
            required_model: 'gemma4:latest',
            model_installed: true,
            ready: true,
            error: null,
        })
        expect(fetch.mock.calls[0][0]).toBe('/api/ollama/dependency-status/')
    })
    it('parses progress records split across streamed chunks', async () => {
        const chunks = [
            '{"status":"downloading","total":100,"completed":2',
            '0}\n{"status":"success"}\n',
        ]
        const body = new ReadableStream<Uint8Array>({
            start(controller) {
                const encoder = new TextEncoder()
                chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
                controller.close()
            },
        })
        const fetch = vi.fn().mockResolvedValue(new Response(body))
        vi.stubGlobal('fetch', fetch)
        const progress: unknown[] = []
        await api.pullOllamaModel((record) => progress.push(record), new AbortController().signal)
        expect(fetch.mock.calls[0][0]).toBe('/api/ollama/pull-model/')
        expect(fetch.mock.calls[0][1].method).toBe('POST')
        expect(progress).toEqual([
            { status: 'downloading', total: 100, completed: 20 },
            { status: 'success' },
        ])
    })
    it('surfaces Ollama stream errors', async () => {
        const fetch = vi.fn().mockResolvedValue(
            new Response('{"error":"disk full"}\n'),
        )
        vi.stubGlobal('fetch', fetch)
        await expect(
            api.pullOllamaModel(vi.fn(), new AbortController().signal),
        ).rejects.toThrow('disk full')
    })
    it('discovers providers and models through the backend', async () => {
        const fetch = vi.fn()
            .mockResolvedValueOnce(new Response(JSON.stringify(['custom'])))
            .mockResolvedValueOnce(new Response(JSON.stringify(['custom-model'])))
        vi.stubGlobal('fetch', fetch)
        expect(await api.providers()).toEqual(['custom'])
        expect(await api.models('custom provider')).toEqual(['custom-model'])
        expect(fetch.mock.calls[0][0]).toBe('/api/get_model_providers/')
        expect(fetch.mock.calls[1][0]).toBe('/api/get_models/?model_provider=custom%20provider')
    })
    it('rejects malformed catalogs and surfaces discovery outages', async () => {
        const fetch = vi.fn()
            .mockResolvedValueOnce(new Response(JSON.stringify({ models: [] })))
            .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Ollama offline' }), { status: 502 }))
        vi.stubGlobal('fetch', fetch)
        await expect(api.providers()).rejects.toThrow('invalid model catalog')
        await expect(api.models('ollama')).rejects.toThrow('Ollama offline')
    })
    it('sends resize in the form and rejects mixed PDF uploads', async () => {
        const fetch = vi.fn().mockResolvedValue(new Response('true'))
        vi.stubGlobal('fetch', fetch)
        await api.upload('pipeline', [new File(['image'], 'page.png')], 'small')
        expect(fetch.mock.calls[0][1].body.get('resize')).toBe('small')
        expect(fetch.mock.calls[0][1].body.getAll('files')).toHaveLength(1)
        await expect(
            api.upload('pipeline', [new File([], 'page.pdf'), new File([], 'page.png')], 'medium'),
        ).rejects.toThrow('one PDF')
    })
    it('fetches generated outputs only and preserves CSV precision', async () => {
        const fetch = vi.fn().mockResolvedValue(new Response('SALDO,Description\n"60.394,59",entry\n'))
        vi.stubGlobal('fetch', fetch)
        const result = await loadResult(
            { data_file_path: 'input.csv', df_file_path: 'plots/result.csv', plot_path: '.' },
            'analyser',
        )
        expect(fetch.mock.calls[0][0]).toBe('/api/df_path/plots/result.csv/download_data')
        expect(result.rows[0][0]).toBe('60.394,59')
        expect(fetch).toHaveBeenCalledTimes(1)
        expect(outputPath('.')).toBeUndefined()
    })
    it('does not fetch the input CSV for plot-only runs', async () => {
        const fetch = vi.fn().mockResolvedValue(new Response('png'))
        vi.stubGlobal('fetch', fetch)
        const result = await loadResult(
            { data_file_path: 'input.csv', plot_path: 'plots/chart.png' },
            'pipeline',
        )
        expect(fetch).toHaveBeenCalledTimes(1)
        expect(result.csv).toBeUndefined()
        expect(result.plot).toBeDefined()
    })
    it('sends scoped retry fields and surfaces agent errors without retries', async () => {
        const fetch = vi
            .fn()
            .mockResolvedValue(new Response(JSON.stringify({ error_message: 'failed' })))
        vi.stubGlobal('fetch', fetch)
        await expect(api.run('analyser', 'plot', 'previous')).rejects.toThrow('failed')
        expect(fetch.mock.calls[0][1].body.get('previous_error')).toBe('previous')
        expect(fetch.mock.calls[0][1].body.get('agent_name')).toBe('analyser')
        expect(fetch).toHaveBeenCalledTimes(1)
    })
})
