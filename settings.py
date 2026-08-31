"""Central paths and environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    source_pdf_dir: Path = PROJECT_ROOT / "data" / "source_pdfs"
    index_root: Path = PROJECT_ROOT / "data" / "faiss_indexes"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 4


SETTINGS = Settings()

