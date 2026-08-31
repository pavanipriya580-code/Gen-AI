"""Command-line helper that builds the two local FAISS indexes."""

from __future__ import annotations

import argparse

from src.rag import build_all_indexes


def main() -> None:
    parser = argparse.ArgumentParser(description="Build academic and fee FAISS indexes.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing indexes after the source PDFs change.",
    )
    args = parser.parse_args()

    results = build_all_indexes(force=args.force)
    print("FAISS index build summary")
    for category, message in results.items():
        print(f"- {category.title()}: {message}")


if __name__ == "__main__":
    main()

