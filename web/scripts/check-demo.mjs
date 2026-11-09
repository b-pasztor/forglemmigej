// CI check of the public demo build: it must be a static site that cannot talk to
// anything. Fails (exit 1) if
//   - index.html lacks the CSP, or the CSP allows network requests or scripts
//     other than the site's own files and its hashed inline bootstrap;
//   - any file mentions the API or the homelab (addresses, host names);
//   - any file links to a host outside the allowlist (attribution, docs);
//   - the license and attribution are missing.
//
// Usage: node scripts/check-demo.mjs build-demo

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { inlineScriptHashes } from './demo-csp.mjs';

const FORBIDDEN = [
	/["'`]\/api\b/, // a call to the real backend
	/\.ts\.net\b/i,
	/\bprodesk\b/i,
	/\btailscale\b/i,
	/\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b/,
	/\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b/, // Tailscale CGNAT range
	/localhost:\d+/,
	/127\.0\.0\.1/,
	/REPORT_GITHUB_TOKEN|POSTGRES_|ghp_|github_pat_/
];

// Links shown to the visitor (sources, licenses, the project) and the URLs
// inside the libraries' own error messages and SVG namespaces.
const ALLOWED_HOSTS = new Set([
	'tatoeba.org',
	'en.wiktionary.org',
	'creativecommons.org',
	'github.com',
	'b-pasztor.github.io',
	'svelte.dev',
	'www.w3.org'
]);

const REQUIRED_TEXT = ['CC BY-SA 4.0', 'CC BY 2.0 FR', 'Wiktionary', 'Tatoeba', 'nothing is sent'];

function files(dir) {
	return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
		entry.isDirectory() ? files(join(dir, entry.name)) : [join(dir, entry.name)]
	);
}

export function checkDemo(dir) {
	const problems = [];
	const html = readFileSync(join(dir, 'index.html'), 'utf8');
	const csp = html.match(/http-equiv="Content-Security-Policy" content="([^"]+)"/)?.[1];
	if (!csp) {
		problems.push('index.html has no Content-Security-Policy');
	} else {
		const directives = Object.fromEntries(
			csp.split(';').map((part) => {
				const [name, ...values] = part.trim().split(/\s+/);
				return [name, values];
			})
		);
		const expect = (name, values) => {
			if (JSON.stringify(directives[name]) !== JSON.stringify(values)) {
				problems.push(
					`CSP ${name} is ${directives[name]?.join(' ')}, expected ${values.join(' ')}`
				);
			}
		};
		expect('default-src', ["'none'"]);
		expect('connect-src', ["'none'"]);
		expect('script-src', ["'self'", ...inlineScriptHashes(html)]);
	}

	const text = [];
	for (const file of files(dir)) {
		if (!/\.(html|js|css|json|txt|svg)$/.test(file)) continue;
		const content = readFileSync(file, 'utf8');
		text.push(content);
		for (const pattern of FORBIDDEN) {
			const match = content.match(pattern);
			if (match) problems.push(`${file}: forbidden ${pattern} (“${match[0]}”)`);
		}
		for (const [, host] of content.matchAll(/https?:\/\/([a-zA-Z0-9.-]+)/g)) {
			if (!ALLOWED_HOSTS.has(host)) problems.push(`${file}: link to ${host} is not allowed`);
		}
	}
	const all = text.join('\n');
	for (const required of REQUIRED_TEXT) {
		if (!all.includes(required)) problems.push(`the build does not mention “${required}”`);
	}
	return problems;
}

if (process.argv[1]?.endsWith('check-demo.mjs')) {
	const dir = process.argv[2] ?? 'build-demo';
	const problems = checkDemo(dir);
	for (const problem of problems) console.error(problem);
	if (problems.length) process.exit(1);
	console.log(`${dir}: static, no API or homelab references, CSP blocks network requests`);
}
