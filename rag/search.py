"""
Semantic search over the Biesse RAG knowledge base.

Flow:

    User query
        ↓
    Gemini Embedding 2
        ↓
    Query embedding
        ↓
    ChromaDB
        ↓
    Top-K relevant chunks

This module does NOT generate answers.
It only retrieves relevant technical documentation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# Configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Add it to the .env file."
    )


EMBEDDING_MODEL = (
    "gemini-embedding-2"
)

EMBEDDING_DIMENSIONS = 1536

COLLECTION_NAME = (
    "biesse_knowledge_base"
)

DEFAULT_TOP_K = 3


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

CHROMA_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "chroma_db"
)


# ============================================================
# Gemini client
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# ChromaDB
# ============================================================

def get_collection():
    """
    Open the existing persistent ChromaDB collection.
    """

    if not CHROMA_DIR.exists():

        raise FileNotFoundError(
            f"ChromaDB directory does not exist:\n"
            f"{CHROMA_DIR}\n\n"
            "Run vector_store.py first."
        )

    chroma_client = (
        chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )
    )

    try:

        collection = (
            chroma_client.get_collection(
                name=COLLECTION_NAME
            )
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not open ChromaDB "
            f"collection '{COLLECTION_NAME}'.\n"
            "Run vector_store.py first."
        ) from error

    return collection


# ============================================================
# Query preparation
# ============================================================

def prepare_query(
    query: str
) -> str:
    """
    Prepare a user query for Gemini Embedding 2.

    The query is formatted specifically for
    question-answering retrieval.
    """

    query = " ".join(query.split())

    return (
        f"task: question answering | query: {query}"
    )


# ============================================================
# Generate query embedding
# ============================================================

def generate_query_embedding(
    query: str
) -> list[float]:
    """
    Generate a 1536-dimensional embedding
    for a user query.
    """

    prepared_query = (
        prepare_query(query)
    )

    response = (
        client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(
                            text=prepared_query
                        )
                    ]
                )
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=(
                    EMBEDDING_DIMENSIONS
                )
            ),
        )
    )

    if not response.embeddings:

        raise RuntimeError(
            "Gemini returned no query embedding."
        )

    embedding = (
        response.embeddings[0].values
    )

    if len(embedding) != (
        EMBEDDING_DIMENSIONS
    ):

        raise RuntimeError(
            f"Expected "
            f"{EMBEDDING_DIMENSIONS}-dimensional "
            f"query embedding, but received "
            f"{len(embedding)}."
        )

    return embedding


# ============================================================
# Search
# ============================================================

def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Perform semantic search against ChromaDB.

    Returns a list of:

        {
            "rank": 1,
            "chunk_id": "...",
            "distance": ...,
            "document": "...",
            "metadata": {...}
        }
    """

    if not query.strip():

        raise ValueError(
            "Search query cannot be empty."
        )

    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )

    collection = (
        get_collection()
    )

    collection_count = (
        collection.count()
    )

    if collection_count == 0:

        raise RuntimeError(
            "ChromaDB collection is empty."
        )

    query_embedding = (
        generate_query_embedding(
            query
        )
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            top_k,
            collection_count,
        ),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = (
        results.get(
            "ids",
            [[]]
        )[0]
    )

    documents = (
        results.get(
            "documents",
            [[]]
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]]
        )[0]
    )

    distances = (
        results.get(
            "distances",
            [[]]
        )[0]
    )

    retrieved = []

    for index, chunk_id in enumerate(
        ids
    ):

        retrieved.append(
            {
                "rank": index + 1,
                "chunk_id": chunk_id,
                "distance": distances[
                    index
                ],
                "document": documents[
                    index
                ],
                "metadata": metadatas[
                    index
                ],
            }
        )

    return retrieved


# ============================================================
# Display search results
# ============================================================

def display_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Pretty-print retrieved chunks.
    """

    print()
    print("=" * 80)
    print("RAG SEMANTIC SEARCH")
    print("=" * 80)

    print()
    print(
        f"Query:\n{query}"
    )

    print()
    print(
        f"Results: {len(results)}"
    )

    for result in results:

        metadata = (
            result["metadata"]
        )

        document = (
            result["document"]
        )

        print()
        print("-" * 80)

        print(
            f"Rank: "
            f"{result['rank']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.6f}"
        )

        print(
            f"Source: "
            f"{metadata.get('source')}"
        )

        print(
            f"Pages: "
            f"{metadata.get('start_page')}-"
            f"{metadata.get('end_page')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section')}"
        )

        print(
            f"Word count: "
            f"{metadata.get('word_count')}"
        )

        print()
        print("Content:")

        print(
            document[:2500]
        )

        if len(document) > 2500:

            print(
                "\n[Content truncated...]"
            )

    print()
    print("=" * 80)


# ============================================================
# Interactive search
# ============================================================

def interactive_search() -> None:
    """
    Start an interactive search session.
    """

    print()
    print("=" * 80)
    print("BIESSE RAG SEARCH")
    print("=" * 80)

    print(
        "Enter a technical support question."
    )

    print(
        "Type 'exit' to quit."
    )

    while True:

        print()

        query = input(
            "Query: "
        ).strip()

        if query.lower() in {
            "exit",
            "quit",
        }:

            print(
                "Exiting."
            )

            break

        if not query:

            continue

        try:

            results = search(
                query,
                top_k=DEFAULT_TOP_K,
            )

            display_results(
                query,
                results,
            )

        except Exception as error:

            print()
            print(
                f"Search failed: {error}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Semantic search over the "
            "Biesse ChromaDB knowledge base."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        help=(
            "Technical support question."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            "Number of results to retrieve "
            "(default: 3)."
        ),
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Start an interactive search "
            "session."
        ),
    )

    args = parser.parse_args()

    if args.interactive:

        interactive_search()

        return

    if not args.query:

        parser.error(
            "Provide a query or use "
            "--interactive."
        )

    results = search(
        args.query,
        top_k=args.top_k,
    )

    display_results(
        args.query,
        results,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()