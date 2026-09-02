"""Deterministic evidence identity for retrieval sufficiency accounting.

This module does not assess corroboration, source reliability, or semantic
similarity. It only groups usable chunks when persisted-record identity can be
proven using exact deterministic signals.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit

from osint_agent.models.document import Document, EvidenceChunk
from osint_agent.models.retrieval import EvidenceGroup, GroupingReason
from osint_agent.storage.sqlite import get_documents_by_ids


_REASON_ORDER: tuple[GroupingReason, ...] = (
    "same_doc_id",
    "canonical_url",
    "content_fingerprint",
    "singleton",
)


def normalize_authoritative_text(text: str) -> str:
    """Normalize full persisted document text for exact identity checks."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def content_fingerprint(text: str) -> str | None:
    """Fingerprint non-empty normalized authoritative text."""

    normalized = normalize_authoritative_text(text)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_url(url: str | None) -> str | None:
    """Conservatively canonicalize a non-empty absolute URL.

    Only scheme/host casing, fragment removal, and trailing-slash handling are
    applied. Query parameters and path spelling remain untouched.
    """

    if not url or not url.strip():
        return None

    try:
        parsed = urlsplit(url.strip())
        if not parsed.scheme or not parsed.hostname:
            return None
        port = parsed.port
    except ValueError:
        return None

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    host = parsed.hostname.lower()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{userinfo}{host}"
    if port is not None:
        netloc += f":{port}"

    path = parsed.path.rstrip("/")
    canonical = SplitResult(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=path,
        query=parsed.query,
        fragment="",
    )
    return urlunsplit(canonical)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _resolved_urls(
    evidence: list[EvidenceChunk],
    documents: Mapping[str, Document],
) -> dict[str, str]:
    urls: dict[str, str] = {}
    chunks_by_doc: dict[str, list[EvidenceChunk]] = defaultdict(list)
    for chunk in evidence:
        chunks_by_doc[chunk.doc_id].append(chunk)

    for doc_id, chunks in chunks_by_doc.items():
        document = documents.get(doc_id)
        if document is not None:
            canonical = canonicalize_url(document.url)
            if canonical is not None:
                urls[doc_id] = canonical
            continue

        # Chroma metadata is a retrieval fallback, not authoritative content.
        candidates = {
            canonical
            for chunk in chunks
            if (canonical := canonicalize_url(chunk.url)) is not None
        }
        if len(candidates) == 1:
            urls[doc_id] = candidates.pop()
    return urls


def _stable_unit_id(chunk_ids: list[str]) -> str:
    payload = "\x1f".join(sorted(chunk_ids)).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return f"evidence-unit-{digest}"


def group_usable_evidence(
    evidence: list[EvidenceChunk],
    authoritative_documents: Mapping[str, Document] | None = None,
) -> list[EvidenceGroup]:
    """Group usable chunks by deterministic persisted-record identity.

    All input chunks remain untouched and available to downstream reasoning.
    Missing authoritative records fail open: absent exact proof, their chunks
    remain separately countable except for same-document or canonical-URL
    identity available from retrieval metadata.
    """

    if not evidence:
        return []

    doc_ids = {chunk.doc_id for chunk in evidence}
    documents = (
        dict(authoritative_documents)
        if authoritative_documents is not None
        else get_documents_by_ids(doc_ids)
    )
    disjoint = _DisjointSet(len(evidence))
    edges: list[tuple[int, int, GroupingReason]] = []
    indexes_by_doc: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(evidence):
        indexes_by_doc[chunk.doc_id].append(index)

    def connect(indexes: list[int], reason: GroupingReason) -> None:
        first = indexes[0]
        for other in indexes[1:]:
            disjoint.union(first, other)
            edges.append((first, other, reason))

    for indexes in indexes_by_doc.values():
        if len(indexes) > 1:
            connect(indexes, "same_doc_id")

    urls_by_doc = _resolved_urls(evidence, documents)
    docs_by_url: dict[str, list[str]] = defaultdict(list)
    for doc_id, url in urls_by_doc.items():
        docs_by_url[url].append(doc_id)
    for matching_docs in docs_by_url.values():
        if len(matching_docs) > 1:
            connect(
                [indexes_by_doc[doc_id][0] for doc_id in sorted(matching_docs)],
                "canonical_url",
            )

    docs_by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for doc_id in sorted(doc_ids):
        document = documents.get(doc_id)
        if document is not None:
            fingerprint = content_fingerprint(document.text)
            if fingerprint is not None:
                docs_by_fingerprint[fingerprint].append(doc_id)
    for matching_docs in docs_by_fingerprint.values():
        if len(matching_docs) > 1:
            connect(
                [indexes_by_doc[doc_id][0] for doc_id in matching_docs],
                "content_fingerprint",
            )

    indexes_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(evidence)):
        indexes_by_root[disjoint.find(index)].append(index)

    reasons_by_root: dict[int, set[GroupingReason]] = defaultdict(set)
    for left, _, reason in edges:
        reasons_by_root[disjoint.find(left)].add(reason)

    manifest = []
    for root, indexes in indexes_by_root.items():
        chunk_ids = sorted(evidence[index].chunk_id for index in indexes)
        reasons = reasons_by_root[root] or {"singleton"}
        manifest.append(
            EvidenceGroup(
                unit_id=_stable_unit_id(chunk_ids),
                member_chunk_ids=chunk_ids,
                member_document_ids=sorted(
                    {evidence[index].doc_id for index in indexes}
                ),
                grouping_reasons=[
                    reason for reason in _REASON_ORDER if reason in reasons
                ],
            )
        )
    return sorted(manifest, key=lambda group: group.unit_id)
