<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type Answer, type AnswerResult, type Next, type Stats } from '../lib/api';
	import { highlight, splitGap } from '../lib/highlight';

	type Phase = 'loading' | 'front' | 'back' | 'feedback' | 'done' | 'error';

	let phase = $state<Phase>('loading');
	let next = $state<Next | null>(null);
	let result = $state<AnswerResult | null>(null);
	let stats = $state<Stats | null>(null);
	let typed = $state('');
	let error = $state('');
	let sessionId: number | null = null;
	let shownAt = 0;
	let responseMs = 0;
	let input = $state<HTMLInputElement | null>(null);

	const card = $derived(next?.card ?? null);
	const isCloze = $derived(card?.card_type === 'cloze');
	const isRecall = $derived(card?.card_type === 'recall');
	const headword = $derived(
		card ? (card.word.article ? `${card.word.article} ${card.word.danish}` : card.word.danish) : ''
	);

	async function load() {
		phase = 'loading';
		result = null;
		typed = '';
		try {
			next = await api.next();
			if (next.card) {
				phase = 'front';
				shownAt = performance.now();
				if (next.card.card_type === 'cloze') setTimeout(() => input?.focus(), 0);
			} else {
				stats = await api.stats();
				phase = 'done';
			}
		} catch (e) {
			error = String(e);
			phase = 'error';
		}
	}

	function reveal() {
		responseMs = performance.now() - shownAt;
		phase = 'back';
	}

	async function submit(answer: Answer) {
		if (!card) return;
		if ('typed_answer' in answer) responseMs = performance.now() - shownAt;
		try {
			result = await api.answer(card.card_id, answer, responseMs, sessionId);
			if (isCloze) {
				phase = 'feedback';
			} else {
				await load();
			}
		} catch (e) {
			error = String(e);
			phase = 'error';
		}
	}

	function addLetter(letter: string) {
		typed += letter;
		input?.focus();
	}

	function formatDue(iso: string | null): string {
		if (!iso) return '';
		return new Date(iso).toLocaleString([], {
			weekday: 'short',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	onMount(async () => {
		try {
			sessionId = (await api.startSession(navigator.userAgent)).id;
		} catch {
			sessionId = null; // reviews still work without a session
		}
		await load();
	});
</script>

<main>
	<header>
		<span class="brand">forglemmigej</span>
		{#if next}
			<span class="counts">{next.due_now} due · {next.new_remaining} new left</span>
		{/if}
	</header>

	{#if phase === 'loading'}
		<p class="muted center">Loading…</p>
	{:else if phase === 'error'}
		<section class="card">
			<p>Something went wrong.</p>
			<p class="muted small">{error}</p>
			<button class="primary" onclick={load}>Try again</button>
		</section>
	{:else if phase === 'done'}
		<section class="card center">
			<h1>All done for now 🌼</h1>
			{#if stats}
				<p>
					Today: {stats.reviewed_today} reviews, {stats.correct_today} remembered, {stats.new_today}
					new.
				</p>
			{/if}
			{#if next?.next_due_at}
				<p class="muted">Next card is due {formatDue(next.next_due_at)}.</p>
			{/if}
			<button class="primary" onclick={load}>Check again</button>
		</section>
	{:else if card}
		<section class="card">
			<p class="kind">
				{#if card.is_new}<span class="badge">new</span>{/if}
				{isCloze ? 'Fill in the Danish word' : isRecall ? 'Say it in Danish' : 'What does it mean?'}
			</p>

			{#if isCloze && card.context}
				{@const [before, after] = splitGap(card.context.danish)}
				<p class="english">{card.context.english}</p>
				<p class="sentence">
					{before}<span class="gap" class:filled={phase === 'feedback'}
						>{phase === 'feedback' ? result?.expected : '___'}</span
					>{after}
				</p>
				{#if phase === 'front'}
					<form
						onsubmit={(e) => {
							e.preventDefault();
							if (typed.trim()) submit({ typed_answer: typed });
						}}
					>
						<input
							bind:this={input}
							bind:value={typed}
							autocomplete="off"
							autocapitalize="off"
							spellcheck="false"
							placeholder="Type the missing word"
						/>
						<div class="letters">
							{#each ['æ', 'ø', 'å'] as letter (letter)}
								<button type="button" onclick={() => addLetter(letter)}>{letter}</button>
							{/each}
						</div>
						<button class="primary wide" type="submit" disabled={!typed.trim()}>Check</button>
					</form>
				{:else if phase === 'feedback' && result}
					<p class={result.correct ? 'verdict good' : 'verdict again'}>
						{result.correct ? 'Correct!' : `You wrote “${typed}”`}
					</p>
					<p class="meaning"><strong>{headword}</strong> = {card.word.english}</p>
					<button class="primary wide" onclick={load}>Next</button>
				{/if}
			{:else}
				{#if isRecall}
					<p class="prompt">{card.word.english}</p>
				{:else if card.context}
					<p class="sentence">
						{#each highlight(card.context.danish, card.context.target_form) as segment, i (i)}
							{#if segment.target}<mark>{segment.text}</mark>{:else}{segment.text}{/if}
						{/each}
					</p>
				{:else}
					<p class="prompt">{headword}</p>
				{/if}

				{#if phase === 'front'}
					<button class="primary wide" onclick={reveal}>Show answer</button>
				{:else}
					<div class="answer">
						{#if card.context}
							<p class="english">{card.context.english}</p>
						{/if}
						<p class="meaning">
							<strong>{headword}</strong>
							{#if card.word.pos}<span class="muted small">({card.word.pos})</span>{/if}
							= {card.word.english}
						</p>
						{#if card.word.compound_parts.length}
							<p class="muted small">{card.word.compound_parts.join(' + ')}</p>
						{/if}
					</div>
					<div class="grades">
						<button class="again" onclick={() => submit({ grade: 1 })}>Didn't know</button>
						<button class="good" onclick={() => submit({ grade: 3 })}>Knew it</button>
					</div>
				{/if}
			{/if}
		</section>
	{/if}
</main>

<style>
	main {
		max-width: 560px;
		margin: 0 auto;
		padding: 16px;
		min-height: 100dvh;
	}
	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		margin-bottom: 16px;
	}
	.brand {
		font-weight: 700;
		color: var(--accent);
	}
	.counts,
	.muted {
		color: var(--muted);
	}
	.small {
		font-size: 0.85rem;
	}
	.center {
		text-align: center;
	}
	.card {
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius);
		padding: 20px;
	}
	.kind {
		margin: 0 0 12px;
		color: var(--muted);
		font-size: 0.9rem;
	}
	.badge {
		background: var(--accent);
		color: var(--accent-text);
		border-radius: 999px;
		padding: 1px 8px;
		margin-right: 6px;
		font-size: 0.75rem;
	}
	.sentence,
	.prompt {
		font-size: 1.5rem;
		line-height: 1.4;
		margin: 8px 0 20px;
	}
	mark {
		background: var(--highlight);
		color: inherit;
		border-radius: 4px;
		padding: 0 3px;
		font-weight: 600;
	}
	.gap {
		display: inline-block;
		min-width: 3ch;
		border-bottom: 2px solid var(--accent);
		font-weight: 600;
	}
	.gap.filled {
		min-width: 0;
	}
	.english {
		color: var(--muted);
		margin: 0 0 8px;
	}
	.meaning {
		margin: 8px 0 16px;
	}
	.answer {
		border-top: 1px solid var(--border);
		padding-top: 12px;
		margin-bottom: 16px;
	}
	input {
		width: 100%;
		font: inherit;
		font-size: 1.25rem;
		padding: 12px;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text);
	}
	.letters {
		display: flex;
		gap: 8px;
		margin: 10px 0 16px;
	}
	.letters button {
		flex: 1;
		padding: 10px;
		font-size: 1.2rem;
		border-radius: 10px;
		border: 1px solid var(--border);
		background: var(--bg);
		color: var(--text);
	}
	button.primary {
		background: var(--accent);
		color: var(--accent-text);
		border: none;
		border-radius: 10px;
		padding: 14px 18px;
		font-weight: 600;
	}
	button:disabled {
		opacity: 0.5;
	}
	.wide {
		width: 100%;
	}
	.grades {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 12px;
	}
	.grades button {
		border: none;
		border-radius: 10px;
		padding: 16px;
		font-weight: 600;
		color: #fff;
	}
	.grades .again {
		background: var(--again);
	}
	.grades .good {
		background: var(--good);
	}
	.verdict {
		font-weight: 600;
	}
	.verdict.good {
		color: var(--good);
	}
	.verdict.again {
		color: var(--again);
	}
</style>
