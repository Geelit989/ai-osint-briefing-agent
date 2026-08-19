"""Opt-in live retrieval -> sufficiency -> local Ollama reasoning smoke test."""

from __future__ import annotations

import argparse
import json

from osint_agent.workflow import generate_brief_for_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--min-evidence", type=int, default=2)
    parser.add_argument("--results", type=int, default=5)
    args = parser.parse_args()

    result = generate_brief_for_query(
        args.query,
        max_distance=args.max_distance,
        min_evidence=args.min_evidence,
        n_results=args.results,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
