// The app talks to one of two backends with the same calls: the real API, or,
// in the public demo build (VITE_DEMO=1), an in-browser one with a fixed deck.
// The demo is loaded with a dynamic import behind a build-time constant, so each
// build contains only its own backend: the demo has no API calls, and the real
// app does not ship the demo deck.

import { api } from './api';
import type { DemoControls } from './demo/backend';

export type Backend = typeof api;
export type { DemoControls };

export const isDemo = import.meta.env.VITE_DEMO === '1';

export async function loadBackend(): Promise<{ backend: Backend; demo: DemoControls | null }> {
	if (isDemo) {
		const { backend, controls } = await import('./demo/backend');
		return { backend, demo: controls };
	}
	return { backend: api, demo: null };
}
