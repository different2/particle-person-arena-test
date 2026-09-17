import { defineConfig } from 'vite';

// Static-friendly build: relative asset paths so `dist/` can be served from anywhere.
export default defineConfig({
  base: './',
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    target: 'es2020',
  },
});
