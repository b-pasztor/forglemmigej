// Add a Content Security Policy to the demo build's index.html (run after
// `vite build --mode demo`). GitHub Pages cannot send headers, so it is a <meta>.
// The policy is the browser-enforced guarantee behind "nothing is sent anywhere":
// connect-src 'none' blocks every fetch/XHR/WebSocket, scripts load only from the
// site itself plus SvelteKit's one inline bootstrap script, allowed by its hash.
//
// Usage: node scripts/demo-csp.mjs build-demo

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export function inlineScriptHashes(html) {
	return [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(
		([, body]) => `'sha256-${createHash('sha256').update(body).digest('base64')}'`
	);
}

export function policy(html) {
	return [
		"default-src 'none'",
		`script-src 'self' ${inlineScriptHashes(html).join(' ')}`,
		"style-src 'self' 'unsafe-inline'",
		"img-src 'self' data:",
		"font-src 'self'",
		"connect-src 'none'",
		"base-uri 'none'",
		"form-action 'none'"
	].join('; ');
}

if (process.argv[1]?.endsWith('demo-csp.mjs')) {
	const file = join(process.argv[2] ?? 'build-demo', 'index.html');
	const html = readFileSync(file, 'utf8');
	if (html.includes('Content-Security-Policy')) throw new Error(`${file} already has a CSP`);
	const meta = `<meta http-equiv="Content-Security-Policy" content="${policy(html)}" />`;
	writeFileSync(
		file,
		html.replace('<meta charset="utf-8" />', `<meta charset="utf-8" />\n\t\t${meta}`)
	);
	console.log(`added CSP to ${file}`);
}
