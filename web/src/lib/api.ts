// Typed client for the forglemmigej API (served under /api; proxied in dev).

export type CardType = 'recognition' | 'recall' | 'cloze' | 'listening' | 'speaking';

export interface Word {
	danish: string;
	article: string | null;
	english: string;
	pos: string | null;
	glosses: string[];
	ipa: string | null;
	compound_parts: string[];
}

export interface Context {
	danish: string; // on cloze cards the target is replaced by "___"
	english: string;
	target_form: string | null; // hidden on cloze cards until answered
}

export interface ReviewCard {
	card_id: number;
	card_type: CardType;
	is_new: boolean;
	word: Word;
	context: Context | null;
	hint: string | null;
}

export interface Next {
	card: ReviewCard | null;
	new_remaining: number;
	practice_waiting: number;
	due_now: number;
	next_due_at: string | null;
}

export interface AnswerResult {
	correct: boolean;
	grade: number;
	expected: string;
	full_sentence: string | null;
	next_due_at: string | null;
	predicted_recall: number | null;
}

export interface Stats {
	reviewed_today: number;
	correct_today: number;
	new_today: number;
	due_now: number;
	due_next_7_days: number;
	cards_by_status: Record<string, number>;
}

export type Answer =
	{ grade: 1 | 3; typed_answer?: never } | { typed_answer: string; grade?: never };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const response = await fetch(`/api${path}`, {
		headers: { 'Content-Type': 'application/json' },
		...init
	});
	if (!response.ok) {
		throw new Error(`${init?.method ?? 'GET'} /api${path} failed: ${response.status}`);
	}
	return response.json() as Promise<T>;
}

export const api = {
	startSession: (device: string) =>
		request<{ id: number }>('/sessions', { method: 'POST', body: JSON.stringify({ device }) }),
	endSession: (id: number) => request<{ id: number }>(`/sessions/${id}/end`, { method: 'POST' }),
	next: () => request<Next>('/next'),
	answer: (cardId: number, answer: Answer, responseTimeMs: number, sessionId: number | null) =>
		request<AnswerResult>(`/cards/${cardId}/answer`, {
			method: 'POST',
			body: JSON.stringify({
				...answer,
				response_time_ms: Math.round(responseTimeMs),
				session_id: sessionId
			})
		}),
	stats: () => request<Stats>('/stats')
};
