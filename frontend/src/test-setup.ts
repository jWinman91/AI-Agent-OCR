import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
Object.defineProperty(window, 'matchMedia', {
  configurable: true,
  value: vi.fn(() => ({ matches: false })),
})
Object.defineProperty(HTMLDialogElement.prototype, 'showModal', {
  configurable: true,
  value: function (this: HTMLDialogElement) {
    this.setAttribute('open', '')
  },
})
Object.defineProperty(HTMLDialogElement.prototype, 'close', {
  configurable: true,
  value: function (this: HTMLDialogElement) {
    this.removeAttribute('open')
  },
})
