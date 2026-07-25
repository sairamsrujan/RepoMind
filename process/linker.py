"""Build the issue <-> PR <-> commit <-> release relationship graph.

Links are derived from:
  * "closes/fixes/resolves #N" in PR bodies and commit messages,
  * GraphQL ``closingIssuesReferences`` on PRs (authoritative),
  * merged-PR -> commit SHAs (from the PR's commits + merge commit),
  * PR -> release, by finding the first release published on/after the PR's
    merge date (the release that first shipped the change).

Output is a JSON graph (``links.json``) used by evaluation and answer context.
"""
from __future__ import annotations

import json
from typing import Any

from core.context import RepositoryContext
from process.chunker import _load_raw, parse_closing_refs


def build_graph(
    commits: list[dict], prs: list[dict], issues: list[dict], releases: list[dict]
) -> dict[str, Any]:
    """Build the relationship graph from raw GitHub records."""
    graph: dict[str, Any] = {
        "issues": {},    # number -> {closed_by_prs, closed_by_commits}
        "prs": {},       # number -> {closes_issues, commits, merged_at, release}
        "commits": {},   # sha    -> {pr, closes_issues}
        "releases": {},  # tag    -> {published_at, prs}
    }

    # Releases first (sorted by date) so PR->release lookup is easy.
    rel_sorted = sorted(
        releases,
        key=lambda r: (r.get("published_at") or r.get("created_at") or ""),
    )
    for r in rel_sorted:
        tag = r.get("tag_name") or r.get("name") or ""
        if tag:
            graph["releases"][tag] = {
                "published_at": r.get("published_at") or r.get("created_at") or "",
                "prs": [],
            }

    def _first_release_after(when: str) -> str | None:
        if not when:
            return None
        for r in rel_sorted:
            pub = r.get("published_at") or r.get("created_at") or ""
            if pub and pub >= when:
                return r.get("tag_name") or r.get("name") or ""
        return None

    # Commits.
    for c in commits:
        sha = c.get("sha", "")
        if not sha:
            continue
        msg = (c.get("commit") or {}).get("message", "")
        closes = parse_closing_refs(msg)
        graph["commits"][sha] = {"pr": None, "closes_issues": closes}
        for n in closes:
            graph["issues"].setdefault(n, {"closed_by_prs": [], "closed_by_commits": []})
            graph["issues"][n]["closed_by_commits"].append(sha)

    # PRs.
    for pr in prs:
        number = pr.get("number")
        if number is None:
            continue
        closing = [n.get("number") for n in
                   (pr.get("closingIssuesReferences", {}) or {}).get("nodes", [])
                   if n.get("number")]
        closing += parse_closing_refs(pr.get("body", "") or "")
        closing = sorted(set(closing))

        shas = [n.get("commit", {}).get("oid", "") for n in
                (pr.get("commits", {}) or {}).get("nodes", [])]
        if pr.get("mergeCommit"):
            shas.append(pr["mergeCommit"].get("oid", ""))
        shas = [s for s in shas if s]

        merged_at = pr.get("mergedAt") or pr.get("merged_at") or ""
        release = _first_release_after(merged_at) if merged_at else None

        graph["prs"][number] = {
            "closes_issues": closing,
            "commits": shas,
            "merged_at": merged_at,
            "release": release,
        }
        for n in closing:
            graph["issues"].setdefault(n, {"closed_by_prs": [], "closed_by_commits": []})
            graph["issues"][n]["closed_by_prs"].append(number)
        for s in shas:
            graph["commits"].setdefault(s, {"pr": None, "closes_issues": []})
            graph["commits"][s]["pr"] = number
        if release and release in graph["releases"]:
            graph["releases"][release]["prs"].append(number)

    return graph


def link_repository(ctx: RepositoryContext) -> dict[str, Any]:
    """Load raw data for ``ctx``, build the graph, and persist ``links.json``."""
    commits = list(_load_raw(ctx.raw_commits_dir))
    prs = list(_load_raw(ctx.raw_prs_dir))
    issues = list(_load_raw(ctx.raw_issues_dir))
    releases = list(_load_raw(ctx.raw_releases_dir))
    graph = build_graph(commits, prs, issues, releases)
    ctx.chunks_dir.mkdir(parents=True, exist_ok=True)
    ctx.links_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return graph


def count_closes_links(graph: dict[str, Any]) -> int:
    """Total number of issue-resolution links (PRs + commits closing issues)."""
    total = 0
    for info in graph.get("issues", {}).values():
        total += len(info.get("closed_by_prs", []))
        total += len(info.get("closed_by_commits", []))
    return total


def to_dot(graph: dict[str, Any], max_issues: int = 12,
           max_commits_per_pr: int = 3) -> str:
    """Render the issue↔PR↔commit↔release graph as Graphviz DOT.

    Only issues that were actually resolved (by a PR or a commit) are shown, so
    the diagram stays readable. Streamlit renders the DOT client-side, so no
    system Graphviz binary is required.
    """
    issues = graph.get("issues", {})
    prs = graph.get("prs", {})
    linked_issues = [
        (num, info) for num, info in issues.items()
        if info.get("closed_by_prs") or info.get("closed_by_commits")
    ][:max_issues]

    lines = [
        "digraph G {",
        '  rankdir=LR; bgcolor="transparent"; pad=0.3; nodesep=0.35; ranksep=0.8;',
        '  node [style="filled,rounded", fontname="Helvetica", fontsize=11,'
        ' color="#2a3242", fontcolor="#e6e8eb"];',
        '  edge [color="#5b6577", fontname="Helvetica", fontsize=9,'
        ' fontcolor="#93a0b4"];',
    ]
    seen: set[str] = set()

    def node(nid: str, label: str, shape: str, fill: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        safe = label.replace('"', "'")
        lines.append(f'  "{nid}" [label="{safe}", shape={shape}, '
                     f'fillcolor="{fill}"];')

    for num, info in linked_issues:
        node(f"issue_{num}", f"Issue #{num}", "box", "#3a201d")
        for p in info.get("closed_by_prs", []):
            node(f"pr_{p}", f"PR #{p}", "ellipse", "#241f4d")
            lines.append(f'  "pr_{p}" -> "issue_{num}" [label="closes"];')
            # prs may be keyed by int (in-memory) or str (loaded from JSON).
            pr_info = prs.get(p) or prs.get(str(p)) or {}
            for sha in pr_info.get("commits", [])[:max_commits_per_pr]:
                node(f"commit_{sha[:12]}", sha[:7], "note", "#12331f")
                lines.append(f'  "pr_{p}" -> "commit_{sha[:12]}";')
            rel = pr_info.get("release")
            if rel:
                node(f"release_{rel}", f"🏷 {rel}", "diamond", "#332a16")
                lines.append(f'  "pr_{p}" -> "release_{rel}" [label="shipped"];')
        for sha in info.get("closed_by_commits", []):
            node(f"commit_{sha[:12]}", sha[:7], "note", "#12331f")
            lines.append(f'  "commit_{sha[:12]}" -> "issue_{num}" [label="fixes"];')

    lines.append("}")
    return "\n".join(lines)
