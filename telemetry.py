"""Phase A: per-query metrics recorder.

Appends one JSON line per answered query to ``config.METRICS_PATH``
(``data/metrics/queries.jsonl``). Recording is best-effort and MUST NEVER crash
or slow the query path: every failure is swallowed silently. Controlled by
``config.ENABLE_METRICS_LOGGING`` — when off, nothing is written and no file is
created.

This module is import-safe for the live Streamlit path: it imports only
``config`` and the standard library, never anything from ``eval/``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import config


def record_query(record: dict[str, Any]) -> None:
    """Append one metrics record (best-effort; never raises).

    The caller supplies the measured fields; this adds a UTC timestamp and the
    current feature-flag state, then writes a single JSON line.
    """
    try:
        if not config.ENABLE_METRICS_LOGGING:
            return
        line = dict(record)
        line.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        line["flags"] = config.flag_state()
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with config.METRICS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - logging must never break a query
        pass
