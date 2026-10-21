# forglemmigej web

The review app: a SvelteKit single-page app (Svelte 5, TypeScript, `adapter-static`,
SSR off). The build is static files, served by nginx in the container, which also
proxies `/api` to the API.

```bash
npm install
npm run dev     # http://localhost:5173, /api proxied to the API on :8000
npm run check   # svelte-check (types)
npm run lint    # prettier + eslint
npm test        # vitest
npm run build   # static site in build/
```

- `src/routes/+page.svelte`: the review screen (recognition cards self-graded, cloze cards typed)
- `src/lib/api.ts`: the API client
- `src/lib/highlight.ts`: highlights the learned word in its sentence
