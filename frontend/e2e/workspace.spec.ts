import { expect, test } from '@playwright/test'

test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname
    if (path.includes('ollama/dependency-status'))
      return route.fulfill({
        json: {
          ollama_reachable: true,
          required_model: 'gemma4:latest',
          model_installed: true,
          ready: true,
          error: null,
        },
      })
    if (path.includes('get_model_providers'))
      return route.fulfill({ json: ['openai', 'ollama'] })
    if (path.includes('get_models'))
      return route.fulfill({
        json: route.request().url().includes('ollama') ? ['gemma4:latest'] : ['gpt-5'],
      })
    if (path.includes('get_config'))
      return route.fulfill({
        json: {
          model: 'gemma4:latest',
          model_provider: 'ollama',
          domain_knowledge_prompt: 'Extract all requested values accurately.',
        },
      })
    if (path.includes('download_data'))
      return route.fulfill({ contentType: 'text/csv', body: 'SALDO,Type\n"60.394,59",Zinsen\n' })
    if (path.includes('download_plot'))
      return route.fulfill({
        contentType: 'image/png',
        body: Buffer.from(
          'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jfHcAAAAASUVORK5CYII=',
          'base64',
        ),
      })
    if (path.includes('run_agents'))
      return route.fulfill({
        json: {
          df_file_path: 'plots/result.csv',
          plot_path: 'plots/result.png',
          code_summary: 'Extracted balances and generated plot.',
        },
      })
    if (path.includes('run_single_agent'))
      return route.fulfill({
        json: {
          data_file_path: 'data/extracted.csv',
          df_file_path: 'plots/analysis.csv',
          plot_path: null,
          code_summary: 'Extracted data.',
        },
      })
    return route.fulfill({ json: true })
  })
})

test('keeps the main workspace blocked until a missing model is confirmed and installed', async ({
  page,
}) => {
  let statusChecks = 0
  await page.route('**/api/ollama/dependency-status/', async (route) => {
    statusChecks += 1
    const installed = statusChecks > 1
    await route.fulfill({
      json: {
        ollama_reachable: true,
        required_model: 'gemma4:latest',
        model_installed: installed,
        ready: installed,
        error: null,
      },
    })
  })
  await page.route('**/api/ollama/pull-model/', async (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/x-ndjson',
      body: '{"status":"downloading","total":100,"completed":100}\n{"status":"success"}\n',
    }),
  )

  await page.goto('/')
  await expect(page.getByRole('dialog', { name: 'Download required model?' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Agent configuration' })).toHaveCount(0)
  await page.getByRole('button', { name: 'Download model' }).click()
  await expect(page.getByRole('heading', { name: 'Agent configuration' })).toBeVisible()
  expect(statusChecks).toBe(2)
})

test('leaves the workspace blocked when the user declines model installation', async ({ page }) => {
  await page.route('**/api/ollama/dependency-status/', async (route) =>
    route.fulfill({
      json: {
        ollama_reachable: true,
        required_model: 'gemma4:latest',
        model_installed: false,
        ready: false,
        error: null,
      },
    }),
  )
  const pulls: string[] = []
  page.on('request', (request) => {
    if (request.url().includes('pull-model')) pulls.push(request.url())
  })

  await page.goto('/')
  await expect(page.getByRole('dialog', { name: 'Download required model?' })).toBeVisible()
  await page.getByRole('button', { name: 'Not now' }).click()
  await expect(page.getByRole('heading', { name: 'Preparing your workspace' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Agent configuration' })).toHaveCount(0)
  expect(pulls).toHaveLength(0)
})

test('configuration, PDF upload, generated outputs and navigation retention', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Agent configuration' })).toBeVisible()
  await expect(page.getByLabel('Provider')).toHaveCount(2)
  await page.getByLabel('Provider').nth(1).selectOption('openai')
  await expect(page.getByLabel('OpenAI API key')).toBeVisible()
  await page.getByRole('link', { name: 'Run AI Agent OCR' }).click()
  await page.locator('input[type=file]').setInputFiles({
    name: 'statement.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('mock PDF'),
  })
  await expect(page.getByRole('dialog', { name: 'Image resolution' })).toBeVisible()
  await page.getByRole('radio', { name: 'Small' }).check()
  const upload = page.waitForRequest((request) => request.url().includes('uploadpdf'))
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  expect((await upload).postData()).toContain('small')
  await page.getByLabel('Analysis prompt').fill('Extract SALDO and plot balances')
  await page.getByRole('button', { name: 'Run pipeline' }).click()
  await expect(page.getByText('60.394,59', { exact: true })).toBeVisible()
  await expect(page.getByAltText('Generated analysis plot')).toBeVisible()
  await expect(page.getByAltText('Generated analysis plot')).toHaveJSProperty('naturalWidth', 1)
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Download CSV' }).click()
  expect((await download).suggestedFilename()).toBe('results.csv')
  await page.getByRole('link', { name: 'Run single agent' }).click()
  await page.getByRole('link', { name: 'Run AI Agent OCR' }).click()
  await expect(page.getByText('60.394,59', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Run pipeline' })).toBeDisabled()
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.screenshot({
    path: `test-results/workspace-${test.info().project.name}.png`,
    fullPage: true,
  })
})

test('single extractor and analyser support CSV-only results', async ({ page }) => {
  await page.goto('/single')
  await page
    .locator('input[type=file]')
    .setInputFiles({ name: 'photo.png', mimeType: 'image/png', buffer: Buffer.from('mock PNG') })
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await page.getByLabel('Extraction prompt').fill('Extract SALDO')
  await page.getByRole('button', { name: 'Run extractor' }).click()
  await expect(page.getByText('60.394,59', { exact: true })).toBeVisible()
  await page.getByRole('tab', { name: 'Analyser', exact: true }).click()
  await page.getByLabel('Analysis prompt').fill('Download external data and calculate totals')
  await page.getByRole('button', { name: 'Run analyser' }).click()
  await expect(page.getByText('60.394,59', { exact: true })).toBeVisible()
  await expect(page.getByAltText('Generated analysis plot')).toHaveCount(0)
})

test('document dialog cancels without uploading and restores file picker focus', async ({
  page,
}) => {
  await page.goto('/single')
  const uploads: string[] = []
  page.on('request', (request) => {
    if (request.url().includes('/upload')) uploads.push(request.url())
  })
  const picker = page.locator('input[type=file]')
  await expect(picker).toBeEnabled()
  const file = { name: 'photo.png', mimeType: 'image/png', buffer: Buffer.from('mock image') }
  await picker.setInputFiles(file)
  await expect(page.getByRole('dialog', { name: 'Image resolution' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(picker).toBeFocused()
  expect(uploads).toHaveLength(0)
  await picker.setInputFiles(file)
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await expect(page.getByText('photo.png', { exact: true })).toBeVisible()
  expect(uploads).toHaveLength(1)
})

test('modern layout, automatic dataset upload, running feedback and split analyser results', async ({
  page,
}, testInfo) => {
  let finish!: () => void
  const pending = new Promise<void>((resolve) => {
    finish = resolve
  })
  await page.route('**/api/run_single_agent', async (route) => {
    await pending
    await route.fulfill({
      json: {
        df_file_path: 'plots/result.csv',
        plot_path: 'plots/chart.png',
        code_summary: 'Analysis complete.',
      },
    })
  })
  await page.goto('/single')
  const selector = await page.getByRole('tablist', { name: 'Agent mode' }).boundingBox()
  const heading = await page
    .getByRole('heading', { name: 'Data extractor', exact: true })
    .boundingBox()
  expect(selector!.y).toBeLessThan(heading!.y)
  await expect(page.getByText('Local workspace', { exact: true })).toHaveCount(0)
  await expect(page.getByRole('heading', { name: 'Results', exact: true })).toHaveCount(0)
  await page.getByRole('tab', { name: 'Analyser', exact: true }).click()
  const documents = await page.locator('.document-pane').boundingBox()
  const prompt = await page.locator('.prompt-pane').boundingBox()
  if (testInfo.project.name === 'mobile') expect(documents!.y).toBeLessThan(prompt!.y)
  else expect(documents!.x + documents!.width).toBeLessThan(prompt!.x)
  const upload = page.waitForRequest((request) => request.url().includes('uploaddatafile'))
  await page
    .locator('input[type=file]')
    .setInputFiles({ name: 'data.csv', mimeType: 'text/csv', buffer: Buffer.from('value\n1\n') })
  await upload
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.getByText('data.csv', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Upload dataset' })).toHaveCount(0)
  await page.getByLabel('Analysis prompt').fill('Calculate totals and plot them')
  await page.getByRole('button', { name: 'Run analyser' }).click()
  await expect(page.locator('.running-status')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Running', exact: true })).toBeDisabled()
  await expect(page.getByRole('heading', { name: 'Results', exact: true })).toHaveCount(0)
  await page.screenshot({
    path: `test-results/running-${testInfo.project.name}.png`,
    fullPage: true,
  })
  finish()
  await expect(page.getByRole('heading', { name: 'Results', exact: true })).toBeInViewport()
  await expect(page.locator('.running-status')).toHaveCount(0)
  const data = await page.locator('.result-data').boundingBox()
  const image = await page.locator('.result-image').boundingBox()
  if (testInfo.project.name === 'mobile')
    expect(data!.y + data!.height).toBeLessThanOrEqual(image!.y)
  else expect(data!.x + data!.width).toBeLessThan(image!.x)
  const inputs = await page.locator('.workflow-grid').boundingBox()
  const results = await page.locator('.results-section').boundingBox()
  expect(inputs!.y + inputs!.height).toBeLessThan(results!.y)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  )
  await page.screenshot({
    path: `test-results/analyser-${testInfo.project.name}.png`,
    fullPage: true,
  })
})
