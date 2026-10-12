import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, loadResult, outputPath } from './api'

afterEach(() => vi.unstubAllGlobals())
describe('backend contracts', () => {
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
