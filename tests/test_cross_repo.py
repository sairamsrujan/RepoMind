"""Phase F tests: cross-repository comparison orchestration (independent indexes)."""
from __future__ import annotations

from types import SimpleNamespace

import config
import query_pipeline
from generation.answerer import extract_citations


class OneRepoRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def retrieve(self, question, **kw):
        self.calls += 1
        return self.chunks


class EchoAnswerer:
    def answer(self, question, chunks, since, until):
        cid = chunks[0]["chunk_id"]
        text = f"Answer from {chunks[0]['github_url']} [{cid}]."
        return SimpleNamespace(text=text, cited_chunk_ids=extract_citations(text))


class GroundedNLI:
    def verify(self, text, chunks):
        return SimpleNamespace(is_grounded=True, contradicted=[], unverified=[])


def test_answer_across_repos_keeps_indexes_independent(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_ADAPTIVE_RETRY", False)
    chunks_a = [{"chunk_id": "pr_1", "source_type": "pr", "text": "A caching",
                 "github_url": "https://github.com/orgA/repoA/pull/1"}]
    chunks_b = [{"chunk_id": "pr_2", "source_type": "pr", "text": "B caching",
                 "github_url": "https://github.com/orgB/repoB/pull/2"}]
    rA, rB = OneRepoRetriever(chunks_a), OneRepoRetriever(chunks_b)
    repos = [("orgA/repoA", rA, "2024-01-01", "2024-12-31"),
             ("orgB/repoB", rB, "2024-01-01", "2024-12-31")]

    results = query_pipeline.answer_across_repos(
        "How was caching handled?", repos, EchoAnswerer(), GroundedNLI())

    # One independent retrieval per repo, in order.
    assert [label for label, _ in results] == ["orgA/repoA", "orgB/repoB"]
    assert rA.calls == 1 and rB.calls == 1

    # Citations stay attributed to their source repository's chunks.
    (_, prA), (_, prB) = results
    assert prA.chunks[0]["github_url"].startswith("https://github.com/orgA")
    assert prB.chunks[0]["github_url"].startswith("https://github.com/orgB")
    assert "orgA/repoA" in prA.text and "orgB/repoB" in prB.text
    assert prA.guard_pass and prB.guard_pass


def test_cross_repo_flag_default_off():
    assert config.ENABLE_CROSS_REPO is False
    assert "ENABLE_CROSS_REPO" in config.flag_state()
