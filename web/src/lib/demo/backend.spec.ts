import { afterEach, describe, expect, it, vi } from 'vitest';
import { checkTypedAnswer, normalizeAnswer } from './answer';
import { createDemo, deck, NEW_WORDS_PER_DAY, type KeyValueStore } from './backend';

const MINUTE = 60_000;
const DAY = 86_400_000;

function memoryStore(): KeyValueStore & { data: Map<string, string> } {
	const data = new Map<string, string>();
	return {
		data,
		getItem: (key) => data.get(key) ?? null,
		setItem: (key, value) => void data.set(key, value),
		removeItem: (key) => void data.delete(key)
	};
}

function setup(store: KeyValueStore | null = memoryStore()) {
	let time = new Date(2026, 10, 9, 9, 0).getTime(); // a morning, local time
	const demo = createDemo(store, () => time);
	return { ...demo, tick: (ms: number) => (time += ms) };
}

afterEach(() => vi.unstubAllGlobals());

describe('the demo deck', () => {
	it('is shared under CC BY-SA with its sources named', () => {
		expect(deck.license.name).toBe('CC BY-SA 4.0');
		expect(deck.attribution.map((a) => a.license)).toEqual(['CC BY-SA 4.0', 'CC BY 2.0 FR']);
	});

	it('links every card to its sources', () => {
		for (const card of deck.cards) {
			const names = card.sources.map((s) => s.name);
			expect(names).toContain('Wiktionary');
			if (card.context) expect(names.filter((n) => n.startsWith('Tatoeba #'))).toHaveLength(2);
			for (const source of card.sources) {
				expect(source.url).toMatch(/^https:\/\/(en\.wiktionary\.org|tatoeba\.org)\//);
			}
		}
	});

	it('holds only deck content, nothing about a learner', () => {
		const allowed = [
			'card_id',
			'card_type',
			'topic',
			'unlocked_by',
			'expected',
			'full_sentence',
			'hint',
			'word',
			'context',
			'sources'
		];
		for (const card of deck.cards) expect(Object.keys(card).sort()).toEqual([...allowed].sort());
	});

	it('never shows the answer on a cloze card', () => {
		for (const card of deck.cards.filter((c) => c.card_type === 'cloze')) {
			expect(card.context?.danish).toContain('___');
			expect(card.context?.target_form).toBeNull();
		}
	});
});

describe('the demo backend', () => {
	it('introduces new words in deck order up to the daily limit', async () => {
		const { backend } = setup();
		const first = await backend.next();
		expect(first.card?.card_id).toBe(deck.cards[0].card_id);
		expect(first.card?.is_new).toBe(true);
		expect(first.new_remaining).toBe(NEW_WORDS_PER_DAY);

		for (let i = 0; i < NEW_WORDS_PER_DAY; i++) {
			const { card } = await backend.next();
			expect(card?.is_new).toBe(true);
			await backend.answer(card!.card_id, { grade: 3 }, 1000, null);
		}
		const after = await backend.next();
		expect(after.new_remaining).toBe(0);
		expect(after.card).toBeNull(); // learning cards are due in a few minutes
		expect(after.next_due_at).not.toBeNull();
	});

	it('predicts recall before the answer, and not for a card seen the first time', async () => {
		const { backend, controls, tick } = setup();
		const { card } = await backend.next();
		const first = await backend.answer(card!.card_id, { grade: 3 }, 1000, null);
		expect(first.predicted_recall).toBeNull();
		tick(10 * MINUTE);
		const graduated = await backend.answer(card!.card_id, { grade: 3 }, 1000, null);
		expect(graduated.predicted_recall).toBeGreaterThan(0.9); // minutes after learning
		const due = +new Date(graduated.next_due_at!);
		tick(due - +controls.now() + 5 * DAY); // well past the due date: memory has faded
		const late = await backend.answer(card!.card_id, { grade: 1 }, 1000, null);
		expect(late.predicted_recall).toBeLessThan(0.9);
	});

	it('unlocks the second card on a later day, once the first one graduated', async () => {
		const { backend, controls, tick } = setup();
		const recognition = deck.cards[0];
		const second = deck.cards.find((c) => c.unlocked_by === recognition.card_id)!;

		await backend.answer(recognition.card_id, { grade: 3 }, 1000, null);
		tick(10 * MINUTE);
		await backend.answer(recognition.card_id, { grade: 3 }, 1000, null);
		expect((await backend.next()).practice_waiting).toBe(0); // same day

		controls.advance(DAY);
		const next = await backend.next();
		expect(next.practice_waiting).toBe(1);
		// Due reviews come first; then the second card takes its turn after a new word.
		const seen: number[] = [];
		for (let i = 0; i < 4; i++) {
			const { card } = await backend.next();
			if (!card) break;
			seen.push(card.card_id);
			await backend.answer(card.card_id, { grade: 3 }, 1000, null);
		}
		expect(seen).toContain(second.card_id);
	});

	it('checks typed answers like the API', async () => {
		const { backend } = setup();
		const cloze = deck.cards.find((c) => c.card_type === 'cloze')!;
		const right = await backend.answer(
			cloze.card_id,
			{ typed_answer: ` ${cloze.expected.toUpperCase()}! ` },
			1000,
			null
		);
		expect(right).toMatchObject({ correct: true, grade: 3, expected: cloze.expected });
		const wrong = await backend.answer(cloze.card_id, { typed_answer: 'nope' }, 1000, null);
		expect(wrong).toMatchObject({ correct: false, grade: 1 });
	});

	it('keeps progress in storage, and starts over on reset or bad data', async () => {
		const store = memoryStore();
		const one = setup(store);
		const { card } = await one.backend.next();
		await one.backend.answer(card!.card_id, { grade: 3 }, 1000, null);
		expect(setup(store).controls.learned()).toBe(1);

		store.data.set([...store.data.keys()][0], '{not json');
		expect(setup(store).controls.learned()).toBe(0);

		one.controls.reset();
		expect(store.data.size).toBe(0);
		expect(setup(null).controls.learned()).toBe(0); // no storage at all
	});

	it('never touches the network', async () => {
		const network = vi.fn(() => {
			throw new Error('network call');
		});
		vi.stubGlobal('fetch', network);
		vi.stubGlobal('XMLHttpRequest', network);
		const { backend, controls } = setup();
		await backend.startSession('test');
		for (let i = 0; i < 30; i++) {
			const { card } = await backend.next();
			if (!card) {
				controls.skipToNextDue();
				continue;
			}
			await backend.answer(card.card_id, { grade: i % 3 ? 3 : 1 }, 1000, null);
		}
		await backend.stats();
		expect(network).not.toHaveBeenCalled();
	});
});

describe('answer normalization', () => {
	// The same cases as the API's normalize_answer (review.py).
	it.each([
		['  Mange tak! ', 'mange tak'],
		['Hvad  så?', 'hvad så'],
		['«Skål»', 'skål'],
		['I GÅR', 'i går']
	])('%s -> %s', (typed, expected) => {
		expect(normalizeAnswer(typed)).toBe(expected);
	});

	it('letters matter', () => {
		expect(checkTypedAnswer('sulten', 'sulten')).toBe(true);
		expect(checkTypedAnswer('tak', 'tåk')).toBe(false);
	});
});
