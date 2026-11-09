// The try-it demo's backend: the same calls as the real API client (lib/api.ts),
// answered in the browser. FSRS-6 with the default parameters schedules a fixed
// deck of survival phrases (deck.json, built by content/demo_deck.py); the
// learner's state stays in this browser's local storage. Nothing is sent anywhere.
//
// The curriculum mirrors the API (review.py): due cards first, then new words up
// to a daily limit; a word's second card unlocks once its first card reached the
// review state and was introduced on an earlier day; new words and second cards
// take turns. A clock offset lets a visitor jump ahead in time.

import {
	createEmptyCard,
	fsrs,
	generatorParameters,
	Rating,
	State,
	type Card as FsrsCard
} from 'ts-fsrs';
import type { Answer, AnswerResult, Next, ReviewCard, Stats } from '../api';
import { checkTypedAnswer } from './answer';
import deckData from './deck.json';

export interface DeckCard extends ReviewCard {
	unlocked_by: number | null;
	expected: string;
	full_sentence: string | null;
	sources: { name: string; url: string }[];
}

export interface Deck {
	license: { name: string; url: string };
	attribution: { name: string; url: string; license: string }[];
	cards: Omit<DeckCard, 'is_new'>[];
}

export const deck = deckData as Deck;
export const NEW_WORDS_PER_DAY = 10;
const STORAGE_KEY = 'forglemmigej-demo-v1';
const DAY_MS = 86_400_000;

// FSRS-6 defaults, as pinned in the API (features.py); no fuzzing, 90% retention.
export const FSRS6_DEFAULT_PARAMETERS = [
	0.212, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.001, 1.8722, 0.1666, 0.796, 1.4835,
	0.0614, 0.2629, 1.6483, 0.6014, 1.8729, 0.5425, 0.0912, 0.0658, 0.1542
];
const scheduler = fsrs(
	generatorParameters({
		w: FSRS6_DEFAULT_PARAMETERS,
		request_retention: 0.9,
		enable_fuzz: false
	})
);

interface StoredCard {
	fsrs: FsrsCard;
	introducedDay: string;
}

interface LogEntry {
	card_id: number;
	at: string;
	grade: number;
	predicted_recall: number | null;
	is_new: boolean;
}

interface State_ {
	offsetMs: number;
	cards: Record<number, StoredCard>;
	log: LogEntry[];
	lastNewWasSecond: boolean;
}

export interface KeyValueStore {
	getItem(key: string): string | null;
	setItem(key: string, value: string): void;
	removeItem(key: string): void;
}

function emptyState(): State_ {
	return { offsetMs: 0, cards: {}, log: [], lastNewWasSecond: true };
}

function revive(state: State_): State_ {
	for (const stored of Object.values(state.cards)) {
		stored.fsrs.due = new Date(stored.fsrs.due);
		if (stored.fsrs.last_review) stored.fsrs.last_review = new Date(stored.fsrs.last_review);
	}
	return state;
}

/** The local calendar day, as the learner experiences it. */
function dayKey(date: Date): string {
	const pad = (n: number) => String(n).padStart(2, '0');
	return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

export function createDemo(store: KeyValueStore | null, clock: () => number = Date.now) {
	let state = load();
	const byId = new Map(deck.cards.map((card) => [card.card_id, card]));

	function load(): State_ {
		try {
			const raw = store?.getItem(STORAGE_KEY);
			return raw ? revive({ ...emptyState(), ...JSON.parse(raw) }) : emptyState();
		} catch {
			return emptyState(); // storage blocked or corrupt: start fresh, in memory
		}
	}

	function save() {
		try {
			store?.setItem(STORAGE_KEY, JSON.stringify(state));
		} catch {
			// private mode or full storage: the demo still works for this visit
		}
	}

	const now = () => new Date(clock() + state.offsetMs);

	function plan(at: Date) {
		const today = dayKey(at);
		const due = deck.cards
			.filter((card) => {
				const stored = state.cards[card.card_id];
				return stored && stored.fsrs.state !== State.New && stored.fsrs.due <= at;
			})
			.sort((a, b) => +state.cards[a.card_id].fsrs.due - +state.cards[b.card_id].fsrs.due);
		const newToday = state.log.filter(
			(entry) =>
				entry.is_new &&
				byId.get(entry.card_id)?.unlocked_by === null &&
				dayKey(new Date(entry.at)) === today
		).length;
		const unseen = deck.cards.filter((card) => !state.cards[card.card_id]);
		const firstCards = unseen.filter((card) => card.unlocked_by === null);
		const secondCards = unseen.filter((card) => {
			const first = card.unlocked_by === null ? null : state.cards[card.unlocked_by];
			return first && first.fsrs.state === State.Review && first.introducedDay < today;
		});
		const newAllowed = Math.max(0, NEW_WORDS_PER_DAY - newToday);
		return {
			due,
			firstCards: firstCards.slice(0, newAllowed),
			secondCards,
			newRemaining: Math.min(newAllowed, firstCards.length)
		};
	}

	function toReviewCard(card: Deck['cards'][number]): ReviewCard {
		const { card_id, card_type, word, context, hint } = card;
		return { card_id, card_type, word, context, hint, is_new: !state.cards[card_id] };
	}

	function nextDueAt(): string | null {
		const dues = Object.values(state.cards)
			.filter((stored) => stored.fsrs.state !== State.New)
			.map((stored) => +stored.fsrs.due);
		return dues.length ? new Date(Math.min(...dues)).toISOString() : null;
	}

	const backend = {
		startSession: async (device: string) => {
			void device;
			return { id: 0 };
		},
		endSession: async (id: number) => ({ id }),
		next: async (): Promise<Next> => {
			const { due, firstCards, secondCards, newRemaining } = plan(now());
			let card = due[0];
			if (!card) {
				const preferSecond = !state.lastNewWasSecond;
				card =
					(preferSecond ? (secondCards[0] ?? firstCards[0]) : (firstCards[0] ?? secondCards[0])) ??
					undefined;
			}
			return {
				card: card ? toReviewCard(card) : null,
				new_remaining: newRemaining,
				practice_waiting: secondCards.length,
				due_now: due.length,
				next_due_at: nextDueAt()
			};
		},
		answer: async (
			cardId: number,
			answer: Answer,
			responseTimeMs: number,
			sessionId: number | null
		): Promise<AnswerResult> => {
			void responseTimeMs;
			void sessionId;
			const card = byId.get(cardId);
			if (!card) throw new Error(`unknown card ${cardId}`);
			const at = now();
			const stored = state.cards[cardId];
			const isNew = !stored;
			const correct =
				answer.typed_answer !== undefined
					? checkTypedAnswer(answer.typed_answer, card.expected)
					: answer.grade > 1;
			const grade = answer.typed_answer !== undefined ? (correct ? 3 : 1) : answer.grade;
			const before = stored?.fsrs ?? createEmptyCard(at);
			// The prediction is made before the answer updates the card, as in the API.
			const predicted =
				before.state === State.New ? null : scheduler.get_retrievability(before, at, false);
			const after = scheduler.next(before, at, grade === 1 ? Rating.Again : Rating.Good).card;
			state.cards[cardId] = { fsrs: after, introducedDay: stored?.introducedDay ?? dayKey(at) };
			if (isNew) state.lastNewWasSecond = card.unlocked_by !== null;
			state.log.push({
				card_id: cardId,
				at: at.toISOString(),
				grade,
				predicted_recall: predicted,
				is_new: isNew
			});
			save();
			return {
				correct,
				grade,
				expected: card.expected,
				full_sentence: card.full_sentence,
				next_due_at: after.due.toISOString(),
				predicted_recall: predicted
			};
		},
		stats: async (): Promise<Stats> => {
			const at = now();
			const today = state.log.filter((entry) => dayKey(new Date(entry.at)) === dayKey(at));
			const byStatus: Record<string, number> = {};
			for (const stored of Object.values(state.cards)) {
				const status = State[stored.fsrs.state].toLowerCase();
				byStatus[status] = (byStatus[status] ?? 0) + 1;
			}
			const week = +at + 7 * DAY_MS;
			return {
				reviewed_today: today.length,
				correct_today: today.filter((entry) => entry.grade > 1).length,
				new_today: today.filter((entry) => entry.is_new).length,
				due_now: plan(at).due.length,
				due_next_7_days: Object.values(state.cards).filter(
					(stored) => stored.fsrs.state !== State.New && +stored.fsrs.due <= week
				).length,
				cards_by_status: byStatus
			};
		}
	};

	const controls = {
		now,
		/** Move the demo's clock forward, e.g. a day, to see cards come back. */
		advance(ms: number) {
			state.offsetMs += ms;
			save();
		},
		/** Jump to the moment the next card is due; false if nothing is scheduled. */
		skipToNextDue(): boolean {
			const due = nextDueAt();
			if (!due) return false;
			state.offsetMs += Math.max(0, +new Date(due) - +now());
			save();
			return true;
		},
		reset() {
			state = emptyState();
			try {
				store?.removeItem(STORAGE_KEY);
			} catch {
				// nothing stored
			}
		},
		/** The model's recall estimate for a card right now, or null if not yet learned. */
		predictedRecall(cardId: number): number | null {
			const stored = state.cards[cardId];
			return stored && stored.fsrs.state !== State.New
				? scheduler.get_retrievability(stored.fsrs, now(), false)
				: null;
		},
		learned: () => Object.keys(state.cards).length,
		total: deck.cards.length,
		deck
	};

	return { backend, controls };
}

function browserStorage(): KeyValueStore | null {
	try {
		return window.localStorage;
	} catch {
		return null;
	}
}

const demo = createDemo(browserStorage());
export const backend = demo.backend;
export const controls = demo.controls;
export type DemoControls = typeof controls;
