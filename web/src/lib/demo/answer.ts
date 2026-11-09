// Typed-answer checking, the same rules as the API (review.py, normalize_answer):
// case, surrounding space and punctuation do not matter; letters do (æ, ø, å).

export function normalizeAnswer(text: string): string {
	return text
		.normalize('NFC')
		.toLowerCase()
		.trim()
		.replace(/[\s.,!?;:"'«»]+/g, ' ')
		.trim();
}

export function checkTypedAnswer(typed: string, expected: string): boolean {
	return normalizeAnswer(typed) === normalizeAnswer(expected);
}
