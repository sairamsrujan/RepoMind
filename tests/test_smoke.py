"""Phase 0 placeholder / smoke tests."""
from __future__ import annotations

import config


def test_config_imports_and_has_constants():
    assert config.SCHEMA_VERSION == 1
    assert config.EMBEDDING_MODEL  # non-empty
    assert config.FINAL_TOP_K > 0
    assert ":latest" not in config.EMBEDDING_MODEL, "no :latest tags allowed"
    assert ":latest" not in config.GENERATION_MODEL, "no :latest tags allowed"


def test_pipeline_fingerprint_shape():
    fp = config.pipeline_fingerprint()
    assert set(fp) == {"schema_version", "embedding_model", "chunker_version"}


def test_judge_model_name_matches_provider():
    # Whatever provider is configured, we get a non-empty model name.
    assert config.judge_model_name()
