#!/usr/bin/env python3
"""
ingest_data.py — Standalone script to load all farming data files
into the ChromaDB vector store.

Run this before starting the backend server (or it runs automatically on startup).

Usage:
    python scripts/ingest_data.py [--reset]

Options:
    --reset    Delete existing collection and re-ingest all data
"""
import argparse
import logging
import sys
import os
from pathlib import Path

# Add backend to path
BACKEND_DIR = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Set default ChromaDB path for local dev
os.environ.setdefault("CHROMA_PERSIST_DIR", str(Path(__file__).parent.parent / "chroma_db"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest farming data into KrishiRakshak ChromaDB vector store"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset ChromaDB collection before ingesting",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    from agent.rag_pipeline import get_vectorstore
    from utils.data_loader import DataLoader

    if args.reset:
        logger.info("Resetting ChromaDB collection...")
        vs = get_vectorstore()
        vs._client.delete_collection("krishirakshak")
        logger.info("Collection deleted.")

    logger.info("Starting data ingestion...")
    loader = DataLoader()

    try:
        from agent.rag_pipeline import get_vectorstore as _vs
        vs = _vs()
        # Force re-load by clearing the global
        import agent.rag_pipeline as rp
        rp._vectorstore = None

        loader.load_all()

        # Verify
        vs = rp.get_vectorstore()
        count = vs._collection.count()
        logger.info(f"✅ Ingestion complete. ChromaDB contains {count} document chunks.")

        # Sample search
        if count > 0:
            logger.info("\n--- Sample Search Test ---")
            results = rp.similarity_search("best crop for Kharif season", k=2)
            for i, doc in enumerate(results, 1):
                logger.info(
                    f"Result {i}: [{doc.metadata.get('source', 'N/A')}] "
                    f"{doc.page_content[:100]}..."
                )

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
