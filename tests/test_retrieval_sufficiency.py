import pytest

from osint_agent.models.document import EvidenceChunk
from osint_agent.retrieval.sufficiency import check_retrieval_sufficiency


def make_evidence(distance: float, index: int = 0) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=f"doc-1::chunk-{index:03}",
        doc_id="doc-1",
        text=f"Evidence text {index}",
        title="Source title",
        source="Example Source",
        provider="example",
        source_type="news",
        published_date="2026-08-15",
        distance=distance,
    )


def test_empty_retrieval_is_insufficient():
    assessment = check_retrieval_sufficiency(
        [], max_distance=0.5, min_evidence=1
    )

    assert assessment.sufficient is False
    assert assessment.reason == "no evidence retrieved"
    assert assessment.evidence_count == 0
    assert assessment.best_distance is None
    assert assessment.usable_evidence == []


def test_single_strong_result_respects_minimum_evidence():
    evidence = [make_evidence(0.31)]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=2
    )

    assert assessment.sufficient is False
    assert assessment.reason == "insufficient usable evidence: 1 < 2"
    assert assessment.usable_evidence == evidence


def test_single_strong_result_is_sufficient_when_minimum_is_one():
    evidence = [make_evidence(0.31)]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=1
    )

    assert assessment.reason == "sufficient evidence: 1 usable chunks"
    assert assessment.evidence_count == 1
    assert assessment.best_distance == 0.31
    assert assessment.usable_evidence == evidence


def test_multiple_strong_results_are_sufficient():
    evidence = [make_evidence(0.31, 0), make_evidence(0.37, 1)]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=2
    )

    assert assessment.evidence_count == 2
    assert assessment.best_distance == 0.31
    assert assessment.usable_evidence == evidence


def test_strong_and_weak_results_only_count_usable_evidence():
    evidence = [
        make_evidence(distance, index)
        for index, distance in enumerate([0.31, 0.37, 0.82, 0.91])
    ]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=2
    )

    assert assessment.sufficient is True
    assert assessment.evidence_count == 4
    assert assessment.usable_evidence == evidence[:2]
    assert assessment.usable_evidence[0] is evidence[0]
    assert assessment.usable_evidence[1] is evidence[1]


def test_all_weak_results_are_insufficient():
    evidence = [
        make_evidence(distance, index)
        for index, distance in enumerate([0.78, 0.84, 0.91])
    ]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=1
    )

    assert assessment.sufficient is False
    assert assessment.reason == "no evidence met relevance threshold"
    assert assessment.best_distance == 0.78


def test_max_distance_threshold_is_inclusive():
    at_threshold = make_evidence(0.5, 0)
    above_threshold = make_evidence(0.5001, 1)

    assessment = check_retrieval_sufficiency(
        [at_threshold, above_threshold],
        max_distance=0.5,
        min_evidence=1,
    )

    assert assessment.sufficient is True
    assert assessment.usable_evidence == [at_threshold]


def test_unsorted_input_uses_lowest_distance_as_best():
    evidence = [
        make_evidence(distance, index)
        for index, distance in enumerate([0.72, 0.31, 0.55, 0.40])
    ]

    assessment = check_retrieval_sufficiency(
        evidence, max_distance=0.5, min_evidence=2
    )

    assert assessment.best_distance == 0.31
    assert assessment.usable_evidence == [evidence[1], evidence[3]]


def test_min_evidence_below_one_is_rejected():
    with pytest.raises(ValueError, match="min_evidence must be at least 1"):
        check_retrieval_sufficiency(
            [make_evidence(0.31)], max_distance=0.5, min_evidence=0
        )
