import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  // GitHub Pages serves the site under the repo path (https://<user>.github.io/<repo>/), so the
  // build needs that prefix in its asset paths. The dev server stays at '/'.
  base: command === 'build' ? '/anime-recommender/' : '/',
  plugins: [react(), tailwindcss()],
}))
