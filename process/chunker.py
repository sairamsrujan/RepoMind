"""Turn raw GitHub data into metadata-tagged, embeddable chunks.

Chunking strategy (see CLAUDE.md Phase 3):
  * commit  -> message headline + body + a *summary* of changed files/stats
              (never the full raw diff — that is too large and noisy to embed).
  * pr      -> title + body + a compact summary of its reviews and linked issues.
  * review  -> one chunk per review (author, state, body).
  * issue   -> title + body.
  * release -> tag/name + release notes body.

Long texts are split on token boundaries so every chunk stays under
``config.MAX_CHUNK_TOKENS``. Each chunk conforms to the schema in CLAUDE.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import config
from core.context import RepositoryContext

# ---- token counting ------------------------------------------------------- #
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))

    def _split_tokens(text: str, max_tokens: int) -> list[str]:
        ids = _ENC.encode(text or "")
        if len(ids) <= max_tokens:
            return [text] if text else []
        return [_ENC.decode(ids[i:i + max_tokens])
                for i in range(0, len(ids), max_tokens)]
except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))

    def _split_tokens(text: str, max_tokens: int) -> list[str]:
        words = (text or "").split()
        if len(words) <= max_tokens:
            return [text] if text else []
        return [" ".join(words[i:i + max_tokens])
                for i in range(0, len(words), max_tokens)]


# ---- reference parsing ---------------------------------------------------- #
# "closes #12", "fixes #7", "resolves #3", or a bare "#42".
_CLOSING_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)
_BARE_ISSUE_RE = re.compile(r"(?<![\w/])#(\d+)\b")


def parse_issue_refs(text: str) -> list[int]:
    """Return all issue/PR numbers referenced in ``text`` (deduped, ordered)."""
    if not text:
        return []
    nums: list[int] = []
    for m in _CLOSING_RE.finditer(text):
        nums.append(int(m.group(1)))
    for m in _BARE_ISSUE_RE.finditer(text):
        nums.append(int(m.group(1)))
    seen: set[int] = set()
    out: list[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse_closing_refs(text: str) -> list[int]:
    """Return only issues explicitly closed/fixed/resolved by ``text``."""
    return sorted({int(m.group(1)) for m in _CLOSING_RE.finditer(text or "")})


# ---- raw loading ---------------------------------------------------------- #
def _load_raw(directory: Path) -> Iterable[dict[str, Any]]:
    """Yield every JSON object from every page/batch file in a raw dir."""
    if not directory.exists():
        return
    for f in sorted(directory.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data


def _make_chunks(base_id: str, source_type: str, ref_id: str, title: str,
                 author: str, date: str, url: str, linked_refs: list,
                 text: str) -> list[dict[str, Any]]:
    """Build one or more schema-conformant chunks, splitting long text."""
    pieces = _split_tokens(text, config.MAX_CHUNK_TOKENS)
    if not pieces:
        pieces = [title or ""]
    chunks = []
    multi = len(pieces) > 1
    for idx, piece in enumerate(pieces):
        chunk_id = f"{base_id}#{idx}" if multi else base_id
        chunks.append({
            "chunk_id": chunk_id,
            "source_type": source_type,
            "ref_id": str(ref_id),
            "title": title or "",
            "author": author or "",
            "date": date or "",
            "github_url": url or "",
            "linked_refs": list(linked_refs),
            "text": piece,
            "token_count": count_tokens(piece),
        })
    return chunks


# ---- per-source builders -------------------------------------------------- #
def _commit_chunks(raw: dict) -> list[dict]:
    sha = raw.get("sha", "")
    commit = raw.get("commit", {})
    message = commit.get("message", "") or ""
    headline = message.splitlines()[0] if message else sha[:12]
    author = (raw.get("author") or {}).get("login") \
        or (commit.get("author") or {}).get("name", "")
    date = (commit.get("author") or {}).get("date", "")
    url = raw.get("html_url", "")

    parts = [message]
    # Optional diff summary: only if a detail fetch populated files/stats.
    files = raw.get("files")
    if files:
        names = ", ".join(f.get("filename", "") for f in files[:config.DIFF_SUMMARY_MAX_LINES])
        parts.append(f"Files changed: {names}")
    stats = raw.get("stats")
    if stats:
        parts.append(f"(+{stats.get('additions', 0)} -{stats.get('deletions', 0)})")
    text = "\n".join(p for p in parts if p)[: config.DIFF_SUMMARY_MAX_CHARS + 4000]

    return _make_chunks(
        base_id=f"commit_{sha[:12]}", source_type="commit", ref_id=sha,
        title=headline, author=author, date=date, url=url,
        linked_refs=parse_issue_refs(message), text=text,
    )


def _pr_chunks(raw: dict) -> list[dict]:
    number = raw.get("number")
    title = raw.get("title", "") or ""
    body = raw.get("body", "") or ""
    author = (raw.get("author") or {}).get("login", "")
    date = raw.get("createdAt") or raw.get("created_at", "")
    url = raw.get("url") or raw.get("html_url", "")

    closing = [n.get("number") for n in
               (raw.get("closingIssuesReferences", {}) or {}).get("nodes", [])]
    commit_shas = [c.get("commit", {}).get("oid", "") for c in
                   (raw.get("commits", {}) or {}).get("nodes", [])]
    if raw.get("mergeCommit"):
        commit_shas.append(raw["mergeCommit"].get("oid", ""))

    # Fold a short review summary into the PR chunk for context.
    reviews = (raw.get("reviews", {}) or {}).get("nodes", [])
    review_summary = "; ".join(
        f"{(r.get('author') or {}).get('login', '?')} {r.get('state', '')}"
        for r in reviews
    )
    body_refs = parse_issue_refs(f"{title}\n{body}")
    linked = sorted({*(n for n in closing if n), *body_refs})

    parts = [f"Pull Request #{number}: {title}", body]
    if closing:
        parts.append("Closes issues: " + ", ".join(f"#{n}" for n in closing if n))
    if review_summary:
        parts.append("Reviews: " + review_summary)
    text = "\n".join(p for p in parts if p)

    chunks = _make_chunks(
        base_id=f"pr_{number}", source_type="pr", ref_id=number, title=title,
        author=author, date=date, url=url,
        linked_refs=linked + [s for s in commit_shas if s], text=text,
    )
    # Each review also becomes its own chunk (source_type="review").
    for i, r in enumerate(reviews):
        rbody = r.get("body", "") or ""
        if not rbody.strip():
            continue
        chunks.extend(_make_chunks(
            base_id=f"review_{number}_{i}", source_type="review", ref_id=number,
            title=f"Review on PR #{number} ({r.get('state', '')})",
            author=(r.get("author") or {}).get("login", ""),
            date=r.get("submittedAt", date),
            url=r.get("url", url), linked_refs=[number], text=rbody,
        ))
    return chunks


def _issue_chunks(raw: dict) -> list[dict]:
    number = raw.get("number")
    title = raw.get("title", "") or ""
    body = raw.get("body", "") or ""
    author = (raw.get("user") or {}).get("login", "")
    date = raw.get("created_at", "")
    url = raw.get("html_url", "")
    text = f"Issue #{number}: {title}\n{body}"
    return _make_chunks(
        base_id=f"issue_{number}", source_type="issue", ref_id=number,
        title=title, author=author, date=date, url=url,
        linked_refs=parse_issue_refs(f"{title}\n{body}"), text=text,
    )


def _release_chunks(raw: dict) -> list[dict]:
    tag = raw.get("tag_name", "") or ""
    name = raw.get("name") or tag
    body = raw.get("body", "") or ""
    author = (raw.get("author") or {}).get("login", "")
    date = raw.get("published_at") or raw.get("created_at", "")
    url = raw.get("html_url", "")
    text = f"Release {name} ({tag})\n{body}"
    return _make_chunks(
        base_id=f"release_{tag or name}".replace(" ", "_"),
        source_type="release", ref_id=tag or name, title=name,
        author=author, date=date, url=url, linked_refs=[], text=text,
    )


# ---- entry point ---------------------------------------------------------- #
def chunk_repository(ctx: RepositoryContext) -> int:
    """Read all raw data for ``ctx`` and write ``chunks.jsonl``. Return count."""
    ctx.chunks_dir.mkdir(parents=True, exist_ok=True)
    builders = [
        (ctx.raw_commits_dir, _commit_chunks),
        (ctx.raw_prs_dir, _pr_chunks),
        (ctx.raw_issues_dir, _issue_chunks),
        (ctx.raw_releases_dir, _release_chunks),
    ]
    count = 0
    with ctx.chunks_path.open("w", encoding="utf-8") as out:
        for directory, builder in builders:
            for raw in _load_raw(directory):
                for chunk in builder(raw):
                    if not chunk["text"].strip():
                        continue
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    count += 1
    return count


def load_chunks(ctx: RepositoryContext) -> list[dict[str, Any]]:
    """Load all chunks written for ``ctx``."""
    if not ctx.chunks_path.exists():
        return []
    chunks = []
    with ctx.chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
