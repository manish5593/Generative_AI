import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path

# Must match the model used by embedding_pipeline.py, or stored vectors and query
# vectors will not be comparable.
EMBEDDING_MODEL = "text-embedding-3-small"

# Longest document snippet included in the context sent to the LLM.
MAX_DOC_CHARS = 1500

# ChromaDB caches clients per path and raises if the same path is reopened in one
# process with different settings, so every client here is built from this one object.
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)


def get_client(chroma_dir: str) -> chromadb.ClientAPI:
    """Open a persistent ChromaDB client with the project's standard settings."""
    return chromadb.PersistentClient(path=str(chroma_dir), settings=CHROMA_SETTINGS)


def _build_embedding_function() -> Optional[OpenAIEmbeddingFunction]:
    """Build the OpenAI embedding function used to embed queries.

    Returns None when no key is configured, letting ChromaDB fall back to its
    default embedder (useful for collections built without OpenAI embeddings).
    """
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    kwargs = {"api_key": api_key, "model_name": EMBEDDING_MODEL}

    # Honour a custom endpoint (e.g. Vocareum) if one is configured.
    api_base = os.getenv("OPENAI_BASE_URL")
    if api_base:
        kwargs["api_base"] = api_base

    return OpenAIEmbeddingFunction(**kwargs)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    chroma_dirs = sorted(d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("chroma"))

    for chroma_dir in chroma_dirs:
        try:
            client = get_client(str(chroma_dir))

            collections = client.list_collections()

            for collection in collections:
                collection_name = collection.name if hasattr(collection, "name") else str(collection)
                key = f"{chroma_dir.name}::{collection_name}"

                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = "unknown"

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{chroma_dir.name} / {collection_name} ({doc_count} docs)",
                    "document_count": str(doc_count),
                }

        except Exception as e:
            # Keep unreadable directories visible so the user can see why they failed.
            error_text = str(e)
            if len(error_text) > 80:
                error_text = error_text[:80] + "..."

            backends[f"{chroma_dir.name}::error"] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (unavailable: {error_text})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = get_client(chroma_dir)

        embedding_function = _build_embedding_function()
        if embedding_function is None:
            return client.get_collection(name=collection_name), True, None

        try:
            collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        except Exception:
            # The collection was persisted with a different embedder (e.g. ChromaDB's
            # default). Its stored configuration wins: queries must be embedded the same
            # way the documents were, or similarity search is meaningless.
            collection = client.get_collection(name=collection_name)

        return collection, True, None

    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", "all missions", ""):
        where_filter = {"mission": mission_filter}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )

    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    context_parts = ["Retrieved NASA mission documents:"]

    for i, (document, metadata) in enumerate(zip(documents, metadatas or [{}] * len(documents)), start=1):
        metadata = metadata or {}

        mission = str(metadata.get("mission", "unknown")).replace("_", " ").title()
        source = metadata.get("source", "unknown")
        category = str(metadata.get("document_category", "unknown")).replace("_", " ").title()

        context_parts.append(f"\n[Source {i}] Mission: {mission} | Source: {source} | Category: {category}")

        if len(document) > MAX_DOC_CHARS:
            context_parts.append(document[:MAX_DOC_CHARS] + "... [truncated]")
        else:
            context_parts.append(document)

    return "\n".join(context_parts)
