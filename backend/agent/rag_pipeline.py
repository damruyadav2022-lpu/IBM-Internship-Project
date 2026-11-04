"""
RAG Pipeline: ChromaDB vector store + sentence-transformers embeddings + retriever.
"""
import json
import logging
import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
# Default to a local folder next to the project root — works on both Windows dev and Docker
_DEFAULT_CHROMA = str(Path(__file__).parent.parent.parent / "chroma_db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", _DEFAULT_CHROMA)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_vectorstore: Chroma | None = None
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return cached sentence-transformer embedding model."""
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    """Return (or create) the ChromaDB vectorstore."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = get_embeddings()
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="krishirakshak",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info(f"ChromaDB loaded from {CHROMA_PERSIST_DIR}")
    return _vectorstore


def get_retriever(k: int = 4):
    """Return a LangChain retriever backed by ChromaDB."""
    vs = get_vectorstore()
    return vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def add_documents(documents: List[Document]) -> None:
    """Add documents to the vectorstore."""
    vs = get_vectorstore()
    vs.add_documents(documents)
    vs.persist()
    logger.info(f"Added {len(documents)} documents to ChromaDB.")


def similarity_search(query: str, k: int = 4) -> List[Document]:
    """Run similarity search and return top-k documents."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)
