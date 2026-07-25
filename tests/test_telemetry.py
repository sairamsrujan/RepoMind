"""Phase A tests: the per-query metrics recorder must be well-formed, silent on
failure, and fully disabled when the flag is off."""
from __future__ import annotations

import json

import config
import telemetry


def test_records_wellformed_line(tmp_path, monkeypatch):
    path = tmp_path / "metrics" / "queries.jsonl"
    monkeypatch.setattr(config, "ENABLE_METRICS_LOGGING", True)
    monkeypatch.setattr(config, "METRICS_DIR", path.parent)
    monkeypatch.setattr(config, "METRICS_PATH", path)

    telemetry.record_query({
        "repo": "acme_widgets", "question": "why?", "empty": False,
        "retrieval_ms": 12.3, "rerank_ms": 45.6, "generation_ms": 789.0,
        "guard_ms": 30.0, "total_ms": 900.0, "guard_verdict": "pass",
        "guard_reason": "ok", "num_citations": 3, "num_valid_citations": 3,
        "num_contradicted": 0, "num_unverified": 1,
        "dense_count": 30, "sparse_count": 30, "rrf_count": 40,
        "mmr_count": 20, "final_count": 6,
    })

    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    # Required fields present.
    for key in ("timestamp", "flags", "repo", "question", "total_ms",
                "guard_verdict", "num_citations", "dense_count", "final_count"):
        assert key in rec, f"missing {key}"
    assert rec["flags"]["ENABLE_METRICS_LOGGING"] is True
    assert rec["repo"] == "acme_widgets"


def test_appends_multiple_lines(tmp_path, monkeypatch):
    path = tmp_path / "metrics" / "queries.jsonl"
    monkeypatch.setattr(config, "ENABLE_METRICS_LOGGING", True)
    monkeypatch.setattr(config, "METRICS_DIR", path.parent)
    monkeypatch.setattr(config, "METRICS_PATH", path)
    for i in range(3):
        telemetry.record_query({"repo": "r", "question": f"q{i}", "total_ms": i})
    assert len(path.read_text().strip().splitlines()) == 3


def test_flag_off_creates_no_file(tmp_path, monkeypatch):
    path = tmp_path / "metrics" / "queries.jsonl"
    monkeypatch.setattr(config, "ENABLE_METRICS_LOGGING", False)
    monkeypatch.setattr(config, "METRICS_DIR", path.parent)
    monkeypatch.setattr(config, "METRICS_PATH", path)
    telemetry.record_query({"repo": "r", "question": "q", "total_ms": 1.0})
    assert not path.exists()
    assert not path.parent.exists()


def test_broken_writer_does_not_raise(tmp_path, monkeypatch):
    # Point the metrics path *under a regular file*, so mkdir/open must fail.
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    bad_path = blocker / "metrics" / "queries.jsonl"
    monkeypatch.setattr(config, "ENABLE_METRICS_LOGGING", True)
    monkeypatch.setattr(config, "METRICS_DIR", bad_path.parent)
    monkeypatch.setattr(config, "METRICS_PATH", bad_path)
    # Must swallow the error silently — never raise into the query path.
    telemetry.record_query({"repo": "r", "question": "q", "total_ms": 1.0})
    assert not bad_path.exists()


def test_flag_state_shape():
    fs = config.flag_state()
    assert "ENABLE_METRICS_LOGGING" in fs
    assert isinstance(fs["ENABLE_METRICS_LOGGING"], bool)
