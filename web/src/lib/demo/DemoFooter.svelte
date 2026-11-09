<script lang="ts">
	import type { DemoControls } from './backend';

	let { demo, cardId, learned }: { demo: DemoControls; cardId: number | null; learned: number } =
		$props();

	const sources = $derived(
		cardId === null ? [] : (demo.deck.cards.find((c) => c.card_id === cardId)?.sources ?? [])
	);
</script>

<footer>
	<p>
		<strong>This is a demo.</strong> It runs entirely in your browser: the schedule, your answers
		and the memory model's estimates stay on this device; nothing is sent anywhere. The real app
		learns from every review on a homelab server; see the
		<a href="https://github.com/b-pasztor/forglemmigej">repository</a> and the
		<a href="https://b-pasztor.github.io/forglemmigej/">live model report</a>.
	</p>
	<p>
		Scheduling: FSRS-6 with its default parameters (via
		<a href="https://github.com/open-spaced-repetition/ts-fsrs">ts-fsrs</a>, MIT), targeting 90%
		recall. Learned {learned} of {demo.total} cards.
	</p>
	{#if sources.length}
		<p>
			This card:
			{#each sources as source, i (source.url)}{i ? ', ' : ''}<a href={source.url}>{source.name}</a
				>{/each}
		</p>
	{/if}
	<p>
		Deck: survival phrases with meanings from
		{#each demo.deck.attribution as source, i (source.url)}{i ? ' and ' : ''}<a href={source.url}
				>{source.name}</a
			>{` (${source.license})`}{/each}, selected and adapted (sentence gaps, first-letter hints).
		The deck is shared under
		<a href={demo.deck.license.url}>{demo.deck.license.name}</a>; the code is MIT.
	</p>
</footer>

<style>
	footer {
		margin-top: 24px;
		color: var(--muted);
		font-size: 0.8rem;
		line-height: 1.5;
	}
	footer p {
		margin: 0 0 8px;
	}
	a {
		color: inherit;
	}
</style>
