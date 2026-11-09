import { defineConfig } from 'vitest/config';
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';

// `vite build --mode demo` builds the public try-it demo (.env.demo sets VITE_DEMO):
// served from GitHub Pages under /forglemmigej/demo/, written to build-demo/.
export default defineConfig(({ mode }) => ({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			// SPA mode: static files, unknown routes fall back to the client-side router.
			adapter: adapter(
				mode === 'demo' ? { pages: 'build-demo', fallback: 'index.html' } : { fallback: '200.html' }
			),
			paths: mode === 'demo' ? { base: '/forglemmigej/demo' } : {}
		})
	],
	server: {
		// In development, forward API calls to the FastAPI backend.
		proxy: { '/api': 'http://localhost:8000' }
	},
	test: {
		expect: { requireAssertions: true },
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'server',
					environment: 'node',
					include: ['src/**/*.{test,spec}.{js,ts}'],
					exclude: ['src/**/*.svelte.{test,spec}.{js,ts}']
				}
			}
		]
	}
}));
