# Environment, Durability & Restore Procedure

RepoMind must still run after a ~9-month gap with no network beyond Ollama on
localhost. This document is the insurance: what can rot, how to detect it, and
how to rebuild on a clean machine.

## What can break in nine months

Ranked by how likely it is to actually happen. Everything marked **observed**
already happened during development — these are not hypotheticals.

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | **A cloud model id is retired** — *observed three times*: `gemini-2.5-flash` began returning *"no longer available"*; several NVIDIA ids 404 despite being listed; Groq retired the answerer `llama-3.3-70b-versatile` on 2026-08-16 with 1 day's notice | **High** | Judge/question-gen fall back to local; scores shift | Chains try the next entry; `check_providers.py` validates every role's model. **Test the replacement before adopting it** — see §"Replacing a retired model" |
| 2 | **A free tier changes or closes** — *observed*: Cerebras returned `402 Payment required` on all models | **High** | That provider drops out | Five providers configured; chain ends at local |
| 3 | **Daily quota exhausted mid-run** — *observed* repeatedly | **Certain** | Evaluation silently changes model | `answered_by` records it; ablation pins one model |
| 4 | **HuggingFace cache cleared** (disk cleanup, new machine) | Medium | Reranker + NLI guard cannot load → **app broken** | `wheelhouse/hf_cache/` (2.9 GB); smoke test loads them in forced-offline mode |
| 5 | **Ollama app auto-updates** and changes model behaviour | Medium | Answers shift subtly | Model **tags** are pinned (never `:latest`); re-run the smoke test after any update |
| 6 | **macOS/Homebrew moves Python** out from under `.venv` | Medium | Nothing runs | Recreate the venv from `wheelhouse/` — no network needed |
| 7 | **A cloud API key is rotated or revoked** | Medium | That provider drops out | Chain + local fallback; `check_providers.py` reports it |
| 8 | **A pinned version is yanked from PyPI** | Low | Clean rebuild fails | 118 wheels in `wheelhouse/` |
| 9 | **GitHub token expires** | **None currently** — this token has no expiry | Would silently drop PRs (GraphQL needs auth) | Smoke test fails at 60 days' notice if one is ever set |
| 10 | **Chroma on-disk format changes** | Low | Indexes unreadable | `chromadb` is pinned; smoke test loads every index |
| 11 | **Disk fills** (indexes + caches ≈ 10 GB) | Low | Obscure failures | Smoke test fails below 5 GB free |

### The single point of failure

**Cloud is optional; local is not.** Every cloud risk above degrades to a local
model. But if Ollama's models or the HuggingFace cache are lost, the app cannot
answer at all — the reranker and NLI guard have no fallback. Those two caches
are the thing to protect:

```
~/.cache/huggingface        6.1 GB   reranker + NLI cross-encoders
ollama models               ~5.4 GB   qwen3-embedding:0.6b, qwen2.5:7b-instruct
wheelhouse/hf_cache         2.9 GB   the backup copy of the above HF models
```

Keep `wheelhouse/` on external storage. It is the difference between a
twenty-minute restore and a broken demo.

### Monthly, and before any demo

```bash
python scripts/smoke_test.py       # 11 checks: environment, models, token, indexes, end-to-end
python scripts/check_providers.py  # cloud providers + evaluation role models
python scripts/demo_check.py       # the things a viewer actually sees
```

`smoke_test.py` is the durability check: it verifies the Python version, free
disk, that every installed package still matches `requirements.txt` exactly,
that Ollama and both pinned tags are present, that the HuggingFace models load
with **networking forced off**, that the GitHub token is valid (warning 60 days
before any expiry), and that all six indexes still load. Run it monthly — the
failures above are all silent until you look.

## Replacing a retired model

Nothing goes down when a cloud model dies: the failure is caught, it is not a
quota error, and generation falls to local `qwen2.5:7b-instruct` with
`fell_back` recorded. So there is time to do this properly.

1. `python scripts/check_providers.py` — lists every provider's live models and
   validates each role.
2. **Test the candidate on citation format before adopting it.** Feed it two
   fake evidence chunks and count the `[bracketed]` tokens in the reply. Groq's
   own recommended replacement for the retired answerer, `qwen/qwen3.6-27b`, is
   a reasoning model: it emitted its `<think>` block and produced 23
   citation-shaped tokens in a two-sentence answer. It would have poisoned the
   reference validator. The vendor's suggestion is not a drop-in.
3. Change `.env` or the chain in `config.py` — never application code.
4. `python -c "import config; print(config.roles_are_distinct())"` — the
   answerer must not become the judge. Groq's other suggestion,
   `openai/gpt-oss-120b`, *is* the judge.
5. Re-run `pytest -q`. Published results are unaffected: each `results.json`
   records `answered_by`, so old numbers stay attributable to the model that
   produced them.

## Freeze (run now, and again before any risky change)

```bash
bash scripts/freeze_environment.sh
```

This:
1. Downloads every pinned wheel to `wheelhouse/` (`pip download -r requirements.txt`).
2. Archives the HuggingFace cache for the reranker and NLI checkpoints into
   `wheelhouse/hf_cache/`.
3. Regenerates the **Snapshot** section below (Python / macOS / `ollama list` /
   `pip freeze`).

`wheelhouse/` is large and **gitignored** — keep it on external storage
(USB/SSD/cloud drive), not in the repo.

## Restore on a clean machine (offline for Python + models)

1. **Python 3.11** and a fresh venv:
   ```bash
   python3.11 -m venv .venv && source .venv/bin/activate
   ```
2. **Install from the wheelhouse** (no network needed for Python deps):
   ```bash
   pip install --no-index --find-links wheelhouse -r requirements.txt
   ```
3. **Restore the model cache** (no HuggingFace download needed):
   ```bash
   mkdir -p ~/.cache/huggingface/hub
   cp -R wheelhouse/hf_cache/* ~/.cache/huggingface/hub/
   ```
4. **Ollama** (the one component that still needs its own install + local models):
   install Ollama, then re-pull the pinned tags:
   ```bash
   ollama pull qwen3-embedding:0.6b
   ollama pull qwen2.5:7b-instruct
   ```
5. **Verify** everything is alive:
   ```bash
   python scripts/smoke_test.py
   ```
   Every check must print `PASS`. The HuggingFace check forces *offline* mode,
   so it fails loudly if a model would need to download — proving the cache
   restore worked.

Indexed repositories under `repositories/` are rebuildable from GitHub and are
gitignored; re-index any you need via the app or `python -m jobs.runner`.

<!-- SNAPSHOT:BEGIN -->
## Snapshot

_Generated 2026-07-24T23:57:44Z_

```
Python: Python 3.11.15
macOS:  ProductName:		macOS ProductVersion:		26.5.2 BuildVersion:		25F84 

ollama list:
NAME                       ID              SIZE      MODIFIED     
qwen2.5:7b-instruct        845dbda0ea48    4.7 GB    2 days ago      
qwen3-embedding:0.6b       ac6da0dfba84    639 MB    2 days ago      
nomic-embed-text:latest    0a109f422b47    274 MB    4 weeks ago     
qwen2.5:0.5b               a8b0c5157701    397 MB    8 weeks ago     
qwen2.5:7b                 845dbda0ea48    4.7 GB    2 months ago    
qwen2.5:3b                 357c53fb659c    1.9 GB    2 months ago    
llama3:latest              365c0bd3c000    4.7 GB    4 months ago    
qwen2:1.5b                 f6daf2b25194    934 MB    5 months ago    

pip freeze:
GitPython==3.1.54
Jinja2==3.1.6
MarkupSafe==3.0.3
PyPika==0.51.1
PyYAML==6.0.3
Pygments==2.20.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.2
aiosignal==1.4.0
altair==6.2.2
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.14.2
attrs==26.1.0
bcrypt==5.0.0
blinker==1.9.0
build==1.5.0
certifi==2026.7.22
charset-normalizer==3.4.9
chromadb==1.5.9
click==8.4.2
distro==1.9.0
durationpy==0.10
filelock==3.32.0
flatbuffers==25.12.19
frozenlist==1.8.0
fsspec==2026.6.0
gitdb==4.0.12
googleapis-common-protos==1.75.0
grpcio==1.82.1
h11==0.16.0
hf-xet==1.5.2
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
huggingface_hub==1.24.0
idna==3.18
importlib_resources==7.1.0
iniconfig==2.3.0
itsdangerous==2.2.0
jiter==0.16.0
joblib==1.5.3
jsonschema-specifications==2025.9.1
jsonschema==4.26.0
kubernetes==36.0.3
markdown-it-py==4.2.0
mdurl==0.1.2
mmh3==5.2.1
mpmath==1.3.0
multidict==6.7.1
narwhals==2.24.0
networkx==3.6.1
numpy==2.4.6
oauthlib==3.3.1
onnxruntime==1.27.0
openai==2.47.0
opentelemetry-api==1.44.0
opentelemetry-exporter-otlp-proto-common==1.44.0
opentelemetry-exporter-otlp-proto-grpc==1.44.0
opentelemetry-proto==1.44.0
opentelemetry-sdk==1.44.0
opentelemetry-semantic-conventions==0.65b0
orjson==3.11.9
overrides==7.7.0
packaging==26.2
pandas==3.0.3
pillow==12.3.0
pluggy==1.6.0
propcache==0.5.2
protobuf==7.35.1
pyarrow==24.0.0
pybase64==1.4.3
pydantic-settings==2.14.2
pydantic==2.13.4
pydantic_core==2.46.4
pydeck==0.9.3
pyproject_hooks==1.2.0
pytest==9.1.1
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
python-multipart==0.0.32
rank-bm25==0.2.2
referencing==0.37.0
regex==2026.7.19
requests-oauthlib==2.0.0
requests==2.34.2
rich==15.0.0
rpds-py==2026.6.3
safetensors==0.8.0
scikit-learn==1.9.0
scipy==1.17.1
sentence-transformers==5.6.0
shellingham==1.5.4
six==1.17.0
smmap==5.0.3
sniffio==1.3.1
starlette==1.3.1
streamlit==1.60.0
sympy==1.14.0
tenacity==9.1.4
threadpoolctl==3.6.0
tiktoken==0.13.0
tokenizers==0.22.2
toml==0.10.2
torch==2.13.0
tqdm==4.69.0
transformers==5.14.1
typer==0.27.0
typing-inspection==0.4.2
typing_extensions==4.16.0
urllib3==2.7.0
uvicorn==0.51.0
uvloop==0.22.1
watchfiles==1.2.0
websocket-client==1.9.0
websockets==16.1.1
yarl==1.24.5
```
<!-- SNAPSHOT:END -->
