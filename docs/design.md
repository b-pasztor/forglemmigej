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
- **Card types per word, easier to harder:** *recognition* (the Danish sentence with the word highlighted → English translation and word meaning), then *cloze* (English sentence + Danish sentence with a gap → type the missing word). The second card (cloze, or recall for words without a sentence) unlocks only once the recognition card has graduated and was introduced on an earlier day, so typing the answer proves something. Words without a validated sentence get plain word cards only if their meaning is unambiguous; others (e.g. *en, et, jo*) are learned through sentences and phrases. Each card type is forgotten separately, so each is a separate card.
- **Phrases show their first letters** on typed and recall cards (*M____ t__*): several phrases can fit one translation ("Thanks": *tak*, *mange tak*), and a miss caused by guessing the wrong synonym would be a false "forgotten" label for the memory model.
- **Nouns always with their article** (*en bil, et hus*); grammatical gender is not predictable.
- **Pacing:** a daily limit of new words (15, configurable). Only a word's first card counts; second cards do not use the limit. New words and second cards take turns, so a backlog of second cards (e.g. after a break) never stops the learner from advancing, and the screen counts them separately ("practice" and "new words"). Rough steady-state load at 90% retention: 15 new words a day means about 90–150 reviews, 15–25 minutes. Planned: reduce new words automatically when the review load is high or retention drops.
- **Planned:** a placement test (a quick "know / don't know" pass over the most common words), and a progress metric: known words among the top 1000 (cards with predicted recall ≥ 90%).

## The memory model (the heart of the project)

For every card the model estimates **"what is the probability that this is recalled
right now?"**. A card is due when that probability would drop below the target
retention (90%). The model therefore makes the daily decision of what to review; it
is not a side report. Better predictions mean fewer reviews for the same retention.

- **Champion: FSRS**, the open-source scheduler used by Anki. A strong baseline while there is little personal data. Fuzzing is off, so scheduling is deterministic and card state can be rebuilt from the review log.
- **Challengers:** (1) FSRS with parameters fitted to the learner's own reviews (`training/fit_fsrs.py`; the optimizer learns only from reviews at least a day after the card's previous one and returns the defaults with fewer than 512 of those, so that is the threshold; on public data, fitted on many reviews, it beat the defaults for every learner); (2) LightGBM pre-trained on the public review logs (`training/pretrain_lightgbm.py`; history features including FSRS's own prediction; on held-out public reviews log loss 0.334 vs 0.405 for default FSRS), in shadow mode from day one so the comparison on personal reviews starts before there is enough data to fit anything; (3) an own model trained on personal data with features FSRS does not see: card type (recognition vs cloze), word vs sentence context, word length, frequency rank, similarity to English/German cognates, response time, time of day; plus elapsed time, review count, lapses and previous grades.
- **Ground truth:** every review is a label (recalled or not), available immediately.
- **Model registry:** `memory_models` holds every model version with its role (champion, challenger, retired) and its exact parameters, so a library upgrade cannot silently change a live model. The champion schedules from the card's cached FSRS state; each challenger predicts every review from the card's history (shadow mode): FSRS challengers by replaying it, LightGBM through the shared feature function. LightGBM models are stored in LightGBM's text format with their feature list and are only challengers: the champion must be FSRS, since scheduling needs a model that can compute the next interval.
- **Champion/challenger gate** (`training/gate.py`, run by the nightly job): a challenger replaces the champion only if it predicts recall better on reviews it has never seen, on **log loss and calibration** together: notebook 02 showed a model can win log loss by hedging while being worse calibrated, and the scheduler acts on the probability around the 90% threshold. The evidence is the shadow predictions since the challenger's registration (it was trained on earlier reviews), on the reviews both models predicted, counting only reviews at least a day after the previous one (minutes after learning every model predicts ~100%). Rules: the challenger can schedule (FSRS); at least 300 such reviews; lower log loss in at least 95% of bootstrap resamples over cards (cards, not reviews, since reviews of one card are not independent); binned calibration RMSE at most 0.01 worse. A fitted FSRS challenger is kept for 14 days so it can collect that evidence, then replaced by a fit on newer data. On promotion the old champion retires, every card's cached state is rebuilt by replaying the log with the new parameters, and the change and its evidence are recorded in `model_promotions` and MLflow. A manual rollback (`gate --promote <version>`) takes the same path.
- **Why the gate is not a CI job:** the review data stays in the homelab (Tailscale only, no public access) and the node is off in the evening and while travelling, so a hosted CI runner could not reach it without opening a path in. The gate runs next to the data, in the nightly job; CI tests the gate's logic.
- **Drift:** forgetting behaviour changes, e.g. after a two-week break, or as the learner improves and new words stick faster. Monitoring must detect it (prediction error over time, input distribution changes).
- **Honest expectation:** with little personal data an own model will likely not beat FSRS at first. The gate rejecting it is the system working as intended. Challengers are pre-trained on a public review-log dataset.
- **Dashboards (Grafana, `deploy/k8s/dashboards/`):** read straight from PostgreSQL by a read-only user, since predictions, reviews and promotions already live there; Prometheus is for system metrics and alerts later. Model health (calibration: predicted vs actual recall; log loss per model version, champion vs shadow challengers on the same reviews; drift alerts; promotion history) and learning progress (known words, reviews per day, retention, 7-day workload forecast).
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
2. **Model report** ([b-pasztor.github.io/forglemmigej](https://b-pasztor.github.io/forglemmigej/)): built by the nightly job after the gate (log loss per model per week, predicted vs actual recall as the drift view, calibration, the gate's evidence in plain words, the registry and promotion history). It shows the system running on real data. Only weekly aggregates leave the homelab: no cards, words, answers or single reviews, and weeks with fewer than 20 usable reviews show only their count. The homelab cannot be reached from outside, so it pushes: `report.json` goes to a separate public data repository with a token that can write only there (a leak could change the report's numbers, never the code), and a Pages workflow publishes it.
3. **Try-it demo** ([b-pasztor.github.io/forglemmigej/demo](https://b-pasztor.github.io/forglemmigej/demo/)): the same review app built in a demo mode (`npm run build:demo`) with a fixed deck: the curated survival phrases with the real card plan (36 phrases, 72 cards). FSRS-6 with the default parameters runs in the browser (ts-fsrs) and the learner's state stays in the browser's local storage; there is no server. The curriculum rules mirror the API's (due first, a daily new-word limit, second cards on a later day, turns). Since real intervals are days long, the visitor can jump ahead in time and sees after each answer what the model expected and when the card returns. The deck is derived from CC BY-SA sources, so it is published under CC BY-SA, and every card links to its Wiktionary entry and Tatoeba sentences.
   - **Nothing can leave the browser, and the build proves it:** the demo backend is a dynamic import behind a build-time constant, so the demo build contains no API client (and the real app no demo deck); a Content Security Policy with `connect-src 'none'` makes the browser block any request; and a CI check (`npm run check:demo`, also run before every Pages publish) fails the build if the CSP is missing or loosened, if any file mentions the API or the homelab (tailnet names, private and Tailscale addresses, tokens), if it links to a host outside a short allowlist, or if the license and attribution are missing. Unit tests run a whole session with `fetch` stubbed to fail.

Order: the core loop first; the model report comes with the retraining work; the demo in the last weeks; the landing page whenever its content is ready.

## Technology choices

- **Frontend:** SvelteKit in SPA mode (Svelte 5 + TypeScript, `adapter-static`, SSR off): small and simple; the build is static files served by nginx.
- **Backend:** Python + FastAPI, packages managed with `uv`; SQLAlchemy 2 + Alembic.
- **Data:** PostgreSQL (items, cards, review log, predictions).
- **ML:** pandas, scikit-learn, LightGBM, the `fsrs` package; Jupyter notebooks for exploration.
- **Experiment tracking:** MLflow (SQLite backend on a volume; one writer is enough). The model registry is the `memory_models` table, since the scheduler reads it on every answer.
- **Orchestration:** Kubernetes CronJobs: `nightly-training` at 19:30, the backup at 20:00 (Prefect or Dagster if it grows).
- **Monitoring:** Grafana on PostgreSQL for model and learning metrics; Prometheus for system metrics and alerting later; drift reports with Evidently.
- **Infrastructure:** Debian 13 + Ansible, single-node k3s; Docker Compose for local development only.
- **CI/CD:** GitHub Actions on hosted runners (`.github/workflows/`): `api` (ruff, the full test suite against PostgreSQL), `web` (svelte-check, lint, tests, build), `deploy` (kustomize + kubeconform, shellcheck, the three image builds); each runs only when its part changes, and one summary check (`ci`) is what branch protection requires. Actions are pinned to commit SHAs and kept current by Dependabot. Deployment stays `deploy.sh` from the laptop, since hosted runners cannot reach the homelab; self-hosted runners must never run pull requests from forks.

## Open questions

- Which Danish TTS voice?
- Cloud LLM for content generation, or local only?
