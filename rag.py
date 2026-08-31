"""PDF loading, chunking, embedding, FAISS indexing, and retrieval."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .settings import SETTINGS


PDF_PATTERNS = {
    "academic": "*academic*.pdf",
    "fee": "*fee*.pdf",
}


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load one small local embedding model and reuse it."""

    return HuggingFaceEmbeddings(
        model_name=SETTINGS.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_category_documents(category: str) -> list[Document]:
    paths = sorted(SETTINGS.source_pdf_dir.glob(PDF_PATTERNS[category]))
    if not paths:
        raise FileNotFoundError(
            f"No {category} PDF found in {SETTINGS.source_pdf_dir}. "
            f"Use a filename matching {PDF_PATTERNS[category]!r}."
        )

    documents: list[Document] = []
    for pdf_path in paths:
        reader = PdfReader(str(pdf_path))
        for page_number, pdf_page in enumerate(reader.pages):
            text = (pdf_page.extract_text() or "").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "category": category,
                            "source_name": pdf_path.name,
                            "page": page_number,
                        },
                    )
                )
    return documents


def build_all_indexes(force: bool = False) -> dict[str, str]:
    """Build one trusted local FAISS index per routed document category."""

    SETTINGS.index_root.mkdir(parents=True, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        length_function=len,
    )
    results: dict[str, str] = {}

    for category in PDF_PATTERNS:
        output_dir = SETTINGS.index_root / category
        if output_dir.exists() and not force:
            results[category] = "kept existing index (use --force to rebuild)"
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)

        pages = _load_category_documents(category)
        chunks = splitter.split_documents(pages)
        vectors = np.asarray(
            get_embeddings().embed_documents([chunk.page_content for chunk in chunks]),
            dtype="float32",
        )
        # Normalized inner product is cosine similarity.
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_dir / "index.faiss"))
        (output_dir / "documents.json").write_text(
            json.dumps(
                [
                    {"page_content": chunk.page_content, "metadata": chunk.metadata}
                    for chunk in chunks
                ],
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        results[category] = f"created {len(chunks)} chunks from {len(pages)} PDF pages"

    load_index.cache_clear()
    return results


@lru_cache(maxsize=2)
def load_index(category: str) -> tuple[object, list[Document]]:
    """Load a FAISS index plus its human-readable document metadata."""

    if category not in PDF_PATTERNS:
        raise ValueError(f"Unsupported retrieval category: {category}")
    index_dir = SETTINGS.index_root / category
    index_path = index_dir / "index.faiss"
    documents_path = index_dir / "documents.json"
    if not index_path.exists() or not documents_path.exists():
        raise FileNotFoundError(
            f"The {category} index is missing. Run `python build_indexes.py` first."
        )
    records = json.loads(documents_path.read_text(encoding="utf-8"))
    documents = [
        Document(page_content=record["page_content"], metadata=record["metadata"])
        for record in records
    ]
    return faiss.read_index(str(index_path)), documents


def retrieve(category: str, query: str, programme: str) -> list[Document]:
    search_text = f"Student programme: {programme}\nQuestion: {query}"
    index, documents = load_index(category)
    query_vector = np.asarray(
        [get_embeddings().embed_query(search_text)], dtype="float32"
    )
    faiss.normalize_L2(query_vector)
    _scores, positions = index.search(query_vector, SETTINGS.top_k)
    return [documents[position] for position in positions[0] if 0 <= position < len(documents)]


def format_context(documents: list[Document]) -> str:
    blocks = []
    for number, document in enumerate(documents, start=1):
        source = document.metadata.get("source_name", Path(str(document.metadata.get("source", "document"))).name)
        page = int(document.metadata.get("page", 0)) + 1
        blocks.append(
            f"[Retrieved passage {number} | {source} | page {page}]\n"
            f"{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def format_sources(documents: list[Document]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    sources: list[str] = []
    for document in documents:
        source = document.metadata.get("source_name", Path(str(document.metadata.get("source", "document"))).name)
        page = int(document.metadata.get("page", 0)) + 1
        key = (str(source), page)
        if key not in seen:
            seen.add(key)
            sources.append(f"{source} (page {page})")
    return sources
