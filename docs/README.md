# Screenshots & demo capture

Captured on **`fastapi/fastapi`** — the most recognisable of the indexed
repositories (5,170 chunks, 2,709 PRs), so a viewer immediately knows the
project being explained.

Drop the files here with these exact names; the README slots are already
written and just need uncommenting.

| File | What it shows |
|---|---|
| `ui-answer.png` | An answered question: citations + green guard badges |
| `ui-refusal.png` | **The important one** — the guard refusing a fabricated premise |
| `ui-graph.png` | The evolution graph (issue ↔ PR ↔ commit ↔ release) |
| `demo.gif` | ~20s recording ending on the refusal |

## Setup

**Turn on adaptive retry first.** Without it a fabricated premise produces a
*soft* refusal — the model declines in prose with amber warning pills. With it,
the guard's widened retry also fails and you get the red **"No verified answer"**
card, which is the screenshot worth having. Add to `.env`:

```bash
ENABLE_ADAPTIVE_RETRY=true
```

Then:

```bash
open -a Ollama                      # the app, not `ollama serve`
.venv/bin/streamlit run app.py
```

At <http://localhost:8501>, paste `fastapi/fastapi` and press **Index / Open**.
It is already indexed, so it loads immediately.

**Ask one throwaway question first and discard it.** The first query loads the
embedding model (~40s); every later one is fast. Recording the cold start makes
the system look slower than it is.

## The three stills

`Cmd-Shift-4`, then `Space`, then click the window.

### 1. `ui-answer.png` — a verified answer

> Why were benchmark tests excluded from the coverage check in PR #14965?

**Verified output:** 1 valid citation, 0 invalid, grounded, guard passed. The
answer cites `pr_14965` and explains it was to speed up coverage processing.

Frame the answer card **and** the badge row beneath it — the badges are the
point, not the prose.

### 2. `ui-refusal.png` — the guard declining

> Why was the `starlette_extras` plugin bundled with FastAPI by default?

No such plugin exists. **Verified output** with adaptive retry on: the guard
rejects the first answer, one widened retry also fails, and you get:

> *I could not find sufficient evidence in the indexed history to answer this
> question with confidence… rather than guess I'm declining to answer.*

Frame it with the question visible so a viewer can see the premise was fabricated.

### 3. `ui-graph.png` — the evolution graph

Expand **Evolution graph** and capture the diagram.

> Note: `fastapi/fastapi` has 127 issue↔PR↔commit links. If the graph looks
> sparse, switch to `pallets/click` (376 links) for this one shot only — a
> denser graph reads better, and nothing else about the shot depends on the repo.

## The GIF

Every other RAG demo ends on a confident answer. Ending on a **refusal** is the
only way to *show* rather than assert that the guard exists — this project's
entire argument, in about five seconds.

Record with [Kap](https://getkap.co) (free) or QuickTime → *New Screen
Recording*, then:

```bash
ffmpeg -i demo.mov -vf "fps=10,scale=1000:-1:flags=lanczos" -loop 0 docs/demo.gif
```

**Sequence — under 25 seconds:**

1. Start with `fastapi/fastapi` already loaded (skip indexing; slow and dull)
2. Ask the **real** question, let the answer land, pause on the green badges (~4s)
3. Clear the box, ask the **fabricated** question
4. Let it refuse — **hold the final frame ~3 seconds and stop there**

Nothing after the refusal: no caption, no cut back to the answer. The refusal is
the ending.

Keep it under ~5 MB so GitHub renders it inline rather than as a download.

## Afterwards

Uncomment the two image blocks in `README.md` (search for `ui-refusal.png` and
`ui-answer.png`), then commit.

Whether you leave `ENABLE_ADAPTIVE_RETRY=true` afterwards is a real choice, not
just a screenshot trick — see the note in the main README's ablation section. It
improved recall 0.607 → 0.684 in the ablation, and it is what makes the honest
refusal visible rather than buried in prose.
