# forglemmigej – design

This document records how the app works and why: the curriculum, the memory
model, the rules that keep the ML honest, the content pipeline and the platform.
The [README](../README.md) is the short version.

## Daily use

From a phone browser, over Tailscale to the homelab; 10–15 minutes a day:

1. The app shows today's due cards, then a few new ones.
2. The learner answers: reads a Danish sentence and checks its meaning, or types a missing Danish word.
3. Recognition cards are self-graded (two buttons); typed answers are checked automatically.
4. In the background the memory model decides when each card comes back.

Planned: audio for every card (Danish text-to-speech) and spoken answers (Whisper).

## Curriculum: what to learn next

Two separate problems: the **curriculum** decides *what* new material to introduce;
the **memory model** decides *when* to review what is already being learned. The
memory model knows nothing about the learner's level; beginner logic lives in the
curriculum.

**Session flow:** due reviews first (memory model), then new cards up to a daily
limit (curriculum); every answer is logged for retraining.

- **Frequency order:** new words come in order of a Danish frequency list (most common first).
- **Survival topics:** curated phrases tagged by topic (greetings, politeness, conversation, time, needs) come first.
- **Words are learned in context, like Duolingo or Babbel.** A learner cannot judge whether a card is correct, so correctness is the pipeline's job, never the learner's. Every word gets one short, human-written Tatoeba example sentence whose English translation confirms the chosen meaning; sentences that fail the check are discarded (see [Content](#content)). Among valid sentences, short ones whose other words are frequent win (i+1: roughly one new thing per card).
- **Card types per word, easier to harder:** *recognition* (the Danish sentence with the word highlighted → English translation and word meaning), then *cloze* (English sentence + Danish sentence with a gap → type the missing word). A cloze card unlocks only once the recognition card has graduated, so the sentence is not still on screen. Words without a validated sentence get plain word cards only if their meaning is unambiguous; others (e.g. *en, et, jo*) are learned through sentences and phrases. Each card type is forgotten separately, so each is a separate card.
- **Nouns always with their article** (*en bil, et hus*); grammatical gender is not predictable.
- **Pacing:** a daily limit for new cards (10). Planned: reduce it when the review load is high or retention drops.
- **Planned:** a placement test (a quick "know / don't know" pass over the most common words), and a progress metric: known words among the top 1000 (cards with predicted recall ≥ 90%).

## The memory model (the heart of the project)

For every card the model estimates **"what is the probability that this is recalled
right now?"**. A card is due when that probability would drop below the target
retention (90%). The model therefore makes the daily decision of what to review; it
is not a side report. Better predictions mean fewer reviews for the same retention.

- **Champion: FSRS**, the open-source scheduler used by Anki. A strong baseline while there is little personal data. Fuzzing is off, so scheduling is deterministic and card state can be rebuilt from the review log.
- **Challengers:** (1) FSRS with parameters fitted to the learner's own reviews (`training/fit_fsrs.py`, from about a thousand reviews; on public data it beat the defaults for every learner); (2) an own model (logistic regression, then LightGBM) with features FSRS does not see: card type (recognition vs cloze), word vs sentence context, word length, frequency rank, similarity to English/German cognates, response time, time of day; plus elapsed time, review count, lapses and previous grades.
- **Ground truth:** every review is a label (recalled or not), available immediately.
- **Model registry:** `memory_models` holds every model version with its role (champion, challenger, retired) and its exact parameters, so a library upgrade cannot silently change a live model. The champion schedules from the card's cached FSRS state; each challenger predicts every review by replaying the card's history (shadow mode).
- **Champion/challenger gate:** retraining runs on a schedule. A challenger replaces the champion only if it predicts recall better on the most recent, chronologically later reviews, on **log loss and calibration** together: notebook 02 showed a model can win log loss by hedging while being worse calibrated, and the scheduler acts on the probability around the 90% threshold. Only reviews at least a day after the previous one count, since every model predicts ~100% minutes after learning. The gate runs in CI.
- **Drift:** forgetting behaviour changes, e.g. after a two-week break, or as the learner improves and new words stick faster. Monitoring must detect it (prediction error over time, input distribution changes).
- **Honest expectation:** with little personal data an own model will likely not beat FSRS at first. The gate rejecting it is the system working as intended. Challengers are pre-trained on a public review-log dataset.
- **Dashboards (Grafana):** model health (calibration: predicted vs actual recall; log loss per model version, champion vs shadow challengers on the same reviews; drift alerts; promotion history) and learning progress (known words, reviews per day, retention, 7-day workload forecast).
- **Stretch:** cold-start difficulty prediction for new words (e.g. compounds of known words are easier), to make the curriculum's pacing smarter.

## Data and ML rules (mandatory)

- **No data leakage:** a prediction for a review may use only information available **before** that review (point-in-time features).
- **Chronological train/test split**, never random shuffling.
- **No train/serve skew:** one shared feature function is used by the scheduler (serving) and by training. A test checks that both paths produce the same features for the same review history.
- **The review log is append-only:** a database trigger forbids UPDATE and DELETE on `reviews`. Card scheduling state is a cache rebuilt from the log.
- **Two-button grading:** Again (1) / Good (3) on the FSRS scale; Hard (2) / Easy (4) stay valid in the schema. The ML label is binary: recalled = grade > 1.
- **Shadow predictions:** for every review, every active model stores its predicted recall in `predictions` (one champion, the others in shadow mode), computed only from the history before that review. Models are compared on exactly the same reviews.
- **Log everything needed for evaluation:** every review stores the card, timestamp, grade, response time, the typed answer, and the predicted recall plus the model version that scheduled it.
- **Scheduler bias:** the champion decides when reviews happen, so the logged data depends on it. Compare models on the same reviews; consider a small share of randomized intervals later.
- **Personal data volume is small** (roughly 50–150 reviews a day). Bootstrap and benchmark challengers on a public review-log dataset (e.g. the open-spaced-repetition Anki review logs; check the license before use) and use personal data for evaluation and fine-tuning.
- **Never click-test against real data:** reviews cannot be deleted, so local previews use a throwaway database.

## Content

- **Verified sources only for core material:** the FrequencyWords Danish list (OpenSubtitles, CC BY-SA 4.0), the Danish Wiktionary via kaikki.org (CC BY-SA 4.0) for meanings, gender, pronunciation and phrases, and Tatoeba Danish–English sentence pairs (CC BY 2.0 FR). Raw data is never committed; attribution is in the README.
- **Word to dictionary entry:** a frequency list counts every surface form ("er", "var", "været" are all *være*). Forms are mapped to their dictionary entry; for frequent words the function-word reading wins over rare homonyms ("for" is a preposition, not "lining"), and bare imperative readings are de-prioritised. Hand-maintained corrections live in `data/curation/word_overrides.toml` (reviewed for the top 500 words); swear words and English subtitle words are excluded.
- **Example validation:** the English translation must contain one of the word's meanings (light stemming; irregular English verbs listed in the curation file). A sentence is rejected if its translation also confirms a *different* word written the same way (*ved* = "at, by", but also "knows" from *vide*); related words (*tak* / *takke*) do not count as competitors. Phrases need a confirming translation or must be the whole sentence ("Mange tak."). Closed-class words (pronouns, articles…) only match their exact form.
- **Small local LLMs are weak in Danish** and may produce ungrammatical sentences a learner cannot spot. An LLM is used only for explanations ("why is the word order like this?"); a stronger cloud API can optionally be used for content generation.
- **Audio (planned):** Danish text-to-speech, local voice to be evaluated.
- **Pronunciation (bonus):** Whisper transcribes a spoken answer and the app compares it with the target. Whisper's Danish is imperfect, so this is a supporting feature, not a main metric.

## Platform

| Machine | Hardware | Role |
|---|---|---|
| HP ProDesk G4 | i5-8500T (6 cores), 16 GB RAM (upgrading to 32 GB, 2×16 GB DDR4 SO-DIMM), 500 GB SSD | **Main node:** Debian 13 + single-node k3s running everything (web, API, PostgreSQL, MLflow, retraining, Prometheus/Grafana; later Whisper, TTS, LLM) |
| HP EliteDesk G3 | i7-7700T (4 cores / 8 threads), 16 GB RAM, 500 GB SSD | Optional: backup target, CI runner, second k3s node later |
| Laptop | Ryzen 5 7535HS, 16 GB RAM | Development, quick experiments |

- **No hypervisor:** Debian on bare metal saves the RAM that a hypervisor and guest OSes would use; k3s does the scheduling.
- **Operating hours:** the node is switched off in the evening and when travelling. Scheduled jobs therefore run before the evening shutdown (database backup at 20:00, Zurich time), and Kubernetes catches up on a missed run at the next boot. The kubelet's graceful node shutdown gives PostgreSQL time to stop cleanly.
- **Memory:** phase 1 (app, database, backups) runs on 16 GB. 32 GB is needed once the LLM and Whisper are added (rough budget: k3s ~1 GB, Postgres/API/web ~1–2 GB, MLflow ~0.5 GB, Prometheus/Grafana ~1 GB, training peak ~2–4 GB, Whisper ~1–2 GB, Qwen3-8B Q4 ~6 GB).
- **Access:** Tailscale only, no public exposure. `tailscale serve` also gives HTTPS on `*.ts.net`, which browsers require for the microphone and for installing the PWA.

## Public showcase (GitHub Pages)

The homelab is switched off while travelling, so everything public is static and
always available:

1. **Personal landing page (later):** a short introduction, contact links, a CV page, and forglemmigej as the featured project. A QR code points to it for conversations at the conference. The design should be modern, clean and Scandinavian, and look professional.
2. **Model report:** generated by the pipeline after each retraining (calibration, champion vs challenger error over time, promotion history, drift events) and published as a static page. It shows the system running on real data. Only aggregated metrics are published, no review-level data.
3. **Try-it demo:** the same review app built in a demo mode with a fixed deck of about 30 survival-phrase cards. FSRS scheduling runs in the browser and answers stay in the browser's local storage; there is no server. The deck is derived from CC BY-SA sources, so it is published under CC BY-SA with attribution.

Order: the core loop first; the model report comes with the retraining work; the demo comes in the last week if time allows; the landing page whenever its content is ready.

## Technology choices

- **Frontend:** SvelteKit in SPA mode (Svelte 5 + TypeScript, `adapter-static`, SSR off): small and simple; the build is static files served by nginx.
- **Backend:** Python + FastAPI, packages managed with `uv`; SQLAlchemy 2 + Alembic.
- **Data:** PostgreSQL (items, cards, review log, predictions).
- **ML:** pandas, scikit-learn, LightGBM, the `fsrs` package; Jupyter notebooks for exploration.
- **Experiment tracking and model registry:** MLflow.
- **Orchestration:** Kubernetes CronJobs for scheduled retraining (Prefect or Dagster if it grows).
- **Monitoring:** Prometheus + Grafana; drift reports with Evidently.
- **Infrastructure:** Debian 13 + Ansible, single-node k3s; Docker Compose for local development only.
- **CI/CD:** GitHub Actions (public repository); self-hosted runners must never run pull requests from forks.

## Open questions

- Which Danish TTS voice?
- Cloud LLM for content generation, or local only?
- GitHub Actions or GitLab CI? (leaning GitHub Actions)
