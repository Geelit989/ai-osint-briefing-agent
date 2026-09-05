from types import SimpleNamespace

from osint_agent.config import settings
from osint_agent.indexing.compatibility import (
    build_compatibility_manifest,
    canonical_manifest_json,
    compatibility_fingerprint,
    manifest_differences,
    resolve_ollama_model_digest,
)


def manifest(**overrides):
    return build_compatibility_manifest(
        model_digest="sha256:model-a",
        embedding_dimension=768,
        overrides=overrides,
    )


def test_identical_and_reordered_manifests_have_identical_fingerprint():
    first = manifest()
    second = dict(reversed(list(first.items())))

    assert canonical_manifest_json(first) == canonical_manifest_json(second)
    assert compatibility_fingerprint(first) == compatibility_fingerprint(second)


def test_material_semantic_changes_each_change_fingerprint():
    baseline = manifest()
    changes = [
        {"embedding_model": "other-model"},
        {"embedding_provider": "changed-ollama-contract"},
        {"embedding_model_digest": "sha256:same-tag-new-content"},
        {"document_embedding_prefix": "different: "},
        {"query_embedding_prefix": "different-query: "},
        {"chunk_size": baseline["chunk_size"] + 1},
        {"chunk_overlap": baseline["chunk_overlap"] + 1},
        {"tokenizer_digest": "sha256:changed-tokenizer"},
        {"text_normalization_version": 2},
        {"index_schema_version": 2},
        {"distance_metric": "cosine"},
    ]

    for change in changes:
        changed = {**baseline, **change}
        assert compatibility_fingerprint(changed) != compatibility_fingerprint(
            baseline
        )
        assert set(manifest_differences(baseline, changed)) == set(change)


def test_irrelevant_application_setting_is_not_in_manifest(monkeypatch):
    before = manifest()
    monkeypatch.setattr(settings, "MAX_RESULTS", 999)
    after = manifest()

    assert "MAX_RESULTS" not in before
    assert before == after
    assert compatibility_fingerprint(before) == compatibility_fingerprint(after)


def test_same_model_tag_resolves_local_content_digest(monkeypatch):
    monkeypatch.setattr(
        "osint_agent.indexing.compatibility.ollama.list",
        lambda: SimpleNamespace(
            models=[
                SimpleNamespace(
                    model="nomic-embed-text:latest", digest="sha256:resolved"
                )
            ]
        ),
    )

    assert resolve_ollama_model_digest("nomic-embed-text") == "sha256:resolved"
