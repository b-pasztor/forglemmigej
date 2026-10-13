import { describe, expect, it } from 'vitest';
import { highlight, splitGap } from './highlight';

describe('highlight', () => {
	it('marks the whole-word target, keeping its original case', () => {
		expect(highlight('Huset er hus.', 'huset')).toEqual([
			{ text: 'Huset', target: true },
			{ text: ' er hus.', target: false }
		]);
	});

	it('does not match inside another word', () => {
		expect(highlight('Erik er her.', 'er')).toEqual([
			{ text: 'Erik ', target: false },
			{ text: 'er', target: true },
			{ text: ' her.', target: false }
		]);
	});

	it('handles Danish letters at word boundaries', () => {
		expect(highlight('Vi er på vej.', 'på')).toEqual([
			{ text: 'Vi er ', target: false },
			{ text: 'på', target: true },
			{ text: ' vej.', target: false }
		]);
	});

	it('returns the plain sentence without a target', () => {
		expect(highlight('Hej!', null)).toEqual([{ text: 'Hej!', target: false }]);
	});
});

describe('splitGap', () => {
	it('splits around the gap', () => {
		expect(splitGap('Vi er ___ vej.')).toEqual(['Vi er ', ' vej.']);
	});
});
