export interface Segment {
	text: string;
	target: boolean;
}

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Split a sentence around the first whole-word occurrence of `form` (case-insensitive). */
export function highlight(sentence: string, form: string | null): Segment[] {
	if (!form) return [{ text: sentence, target: false }];
	const pattern = new RegExp(`(?<![\\p{L}\\p{N}])${escape(form)}(?![\\p{L}\\p{N}])`, 'iu');
	const match = pattern.exec(sentence);
	if (!match) return [{ text: sentence, target: false }];
	const start = match.index;
	const end = start + match[0].length;
	return [
		{ text: sentence.slice(0, start), target: false },
		{ text: match[0], target: true },
		{ text: sentence.slice(end), target: false }
	].filter((segment) => segment.text !== '');
}

/** Split a cloze sentence around its gap ("___"). */
export function splitGap(sentence: string): [string, string] {
	const index = sentence.indexOf('___');
	return index < 0 ? [sentence, ''] : [sentence.slice(0, index), sentence.slice(index + 3)];
}
