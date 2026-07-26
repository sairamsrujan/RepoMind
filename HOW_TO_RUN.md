# How to Run RepoMind

Everything you need to start the app, verify it works, and fix it if it doesn't.
Written to still make sense months from now.

---

## 1. Quick start (if it's already set up on this machine)

```bash
cd "/Users/rsairamsrujankumar/Projects/FINAL YEAR PROJECT/RepoMind"
open -a Ollama
.venv/bin/streamlit run app.py
```

Then open **http://localhost:8501**.

To stop the app: press `Ctrl+C` in the terminal, or run `pkill -f "streamlit run app.py"`.

---

## 2. Before a demo — always run this first

```bash
cd "/Users/rsairamsrujankumar/Projects/FINAL YEAR PROJECT/RepoMind"
.venv/bin/python scripts/demo_check.py
```

It exercises exactly what an evaluator will see and prints `DEMO READY ✅` or
tells you what is broken:

| Check | Why it matters in the demo |
|---|---|
| Ollama + models present | The app cannot answer without them |
| Indexed repositories load | You need something to show |
| Grounded answer + real citations | The citations must be clickable |
| Guard catches a fake citation | Your headline "it can't hallucinate" moment |
| Refuses a made-up premise | Proves it says "I don't know" |
| Flags out-of-coverage dates | Proves it knows its own limits |
| Evolution graph has links | The visual everyone reacts to |
| Cloud works **and** falls back | Proves the demo survives bad wifi |

**Run this before every presentation.** It takes a couple of minutes and removes
the risk of finding out live that something broke.

---

## 3. Monthly health check (during the months before the final evaluation)

```bash
cd "/Users/rsairamsrujankumar/Projects/FINAL YEAR PROJECT/RepoMind"
.venv/bin/python scripts/smoke_test.py
```

This is the long-term insurance. It additionally confirms the reranker and NLI
models still load **offline** from the local cache and runs the full test suite.
If it says `ALL CHECKS PASSED`, the project is healthy.

Put a monthly reminder in your calendar. This one habit is what protects you
against environment drift over nine months.

---

## 4. Using the app

1. Paste a public GitHub repo URL (e.g. `https://github.com/psf/requests`) or
   just `owner/name`, set the months window, click **Index / Open**.
2. Already-indexed repos load instantly (the index is reused, not rebuilt).
   Switch between them from the sidebar.
3. Ask a question. You get an answer with inline citations that link to the real
   commits, PRs, and issues, plus guard badges showing whether the answer was
   verified.
4. Panels available: **stats**, **evolution graph** (issue ↔ PR ↔ commit ↔
   release), **evaluation metrics**, **recent questions**, and
   **export answer to Markdown**.

### Repos currently indexed

| Repo | Good for showing |
|---|---|
| `pallets/click` | Rich evolution graph (570 PRs, 376 links) |
| `psf/requests` | Well-known project, 658 PRs |
| `fastapi/fastapi` | Large corpus (1,885 chunks) |
| `acme/widgets` | Tiny synthetic repo — fast, predictable answers |

---

## 5. First-time setup (new machine, or if `.venv` is missing)

```bash
cd "/Users/rsairamsrujankumar/Projects/FINAL YEAR PROJECT/RepoMind"
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), then pull the two pinned models:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5:7b-instruct
```

Create `.env` in the project root (it is gitignored — never commit it):

```
GITHUB_TOKEN=your_github_token_here
```

`GITHUB_TOKEN` needs only the **`public_repo`** scope. Without it you get
60 requests/hour and **no pull requests at all**, because PR ingestion uses the
GraphQL API, which requires authentication.

**Offline install:** if you have the `wheelhouse/` folder from
`scripts/freeze_environment.sh`, you can install with no network:
`.venv/bin/pip install --no-index --find-links wheelhouse -r requirements.txt`.
See `ENVIRONMENT.md` for the full restore procedure.

---

## 6. Optional: cloud generation (better answers, needs internet)

By default everything runs **locally** — no internet, nothing can fail. To use a
hosted model instead, add to `.env`:

```
GENERATION_PROVIDER=api
GENERATION_API_BASE_URL=https://api.groq.com/openai/v1
GENERATION_API_MODEL=llama-3.3-70b-versatile
GENERATION_API_KEY=your_groq_key_here
JUDGE_PROVIDER=groq
GROQ_API_KEY=your_groq_key_here
```

If the API fails for **any** reason — bad key, rate limit, retired model, no
wifi — the app automatically answers with the local model instead and shows a
`☁️→💻 cloud unavailable, answered locally` badge. The demo cannot go down
because a provider did.

To go back to fully local, just delete or comment out `GENERATION_PROVIDER=api`.

> **Safer for a live demo:** run **local** (`GENERATION_PROVIDER` unset). Zero
> external dependencies. Use the API for offline evaluation runs instead.

---

## 7. Running tests and evaluation

```bash
.venv/bin/python -m pytest -q                       # full test suite
.venv/bin/python -m eval.run --repo pallets/click \
    --dataset eval/datasets/pallets_click.jsonl \
    --out results/my-run                            # evaluation
.venv/bin/python scripts/summarise_metrics.py       # per-query latency/guard stats
```

Evaluations **checkpoint after every question**. If one is interrupted, re-run
the exact same command and it resumes where it stopped.

---

## 8. ⚠️ Important: memory (16 GB machines)

This is the single most common cause of trouble, so read it before running
anything heavy.

The models are large: the chat model is ~4.7 GB, embeddings ~2.4 GB, reranker
~2.3 GB, NLI guard ~0.8 GB. **Run only ONE heavy job at a time.**

**Do not** run the app, `pytest`, and an evaluation simultaneously. On 16 GB the
machine starts swapping, becomes very laggy and hot, and the OS may kill a
process **with no error message at all** — a job that stops silently is almost
always this.

Free Ollama's models between jobs:

```bash
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5:7b-instruct","keep_alive":0}'
```

Check available memory:

```bash
memory_pressure | tail -1
```

For long evaluations, either use the cloud API (moves the biggest model off your
machine) or the smaller local model with `GENERATION_MODEL=qwen2.5:3b`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A job stops with no error | Out of memory | Close other apps; run one job at a time (§8) |
| Laptop lagging / hot | Too many jobs at once | Same as above |
| "Could not reach Ollama" | Ollama not running | `open -a Ollama` |
| PRs show 0 for a repo | Indexed without a valid token | Fix `GITHUB_TOKEN`, then re-index |
| "Bad credentials" | Token invalid/expired | Generate a new one (`public_repo` scope) |
| Answers slow (~10s) | Normal — reranker + guard | Expected; mention it as a design tradeoff |
| First query very slow | Models loading into RAM | Ask one warm-up question before demoing |
| Cloud badge shows fallback | API rate-limited or down | Working as designed; local answered instead |
| Port 8501 already in use | App already running | `pkill -f "streamlit run app.py"` |

### Reset a repository's index

```bash
rm -rf "repositories/<owner>_<name>"
```

Then re-index it from the app.

---

## 10. Demo script (suggested 5-minute flow)

1. **Warm up first** — ask one question before the audience is watching, so
   models are already in RAM and the first live answer is fast.
2. Open `pallets/click` from the sidebar — show the stats panel.
3. Expand the **evolution graph** — 376 real issue↔PR↔commit links.
4. Ask: *"Why was isolated_filesystem deprecated?"* — show the answer, then
   **click a citation** to open the real GitHub page.
5. Point at the **guard badges** — explain citations were verified against the
   retrieved evidence.
6. Ask something fabricated: *"Why was the blockchain module removed?"* — it
   refuses instead of inventing. **This is the moment that lands.**
7. Mention: everything runs locally; the cloud model is optional and falls back
   automatically.

Keep `scripts/demo_check.py` output from earlier that day as proof it was
verified working.
