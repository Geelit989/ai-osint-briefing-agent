"""Run a live Currents search and print one normalized document."""

from osint_agent.ingestion.currents import search_currents


def main() -> None:
    documents = search_currents(
        query="China",
        limit=1,
    )

    if not documents:
        print("No usable documents returned.")
        return

    print(documents[0].model_dump_json(indent=2))


if __name__ == "__main__":
    main()