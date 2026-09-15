"""
ChromaDB vector store for the Biesse RAG system.

Input:
    knowledge_base/processed/embeddings.jsonl

Output:
    knowledge_base/chroma_db/

Responsibilities:
    - Create/open persistent ChromaDB collection
    - Load Gemini embeddings
    - Store embeddings, documents and metadata
    - Avoid duplicate inserts
    - Validate collection size
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "processed"
)

EMBEDDINGS_FILE = (
    PROCESSED_DIR
    / "embeddings.jsonl"
)

CHROMA_DIR = (
    PROJECT_ROOT
    / "knowledge_base"
    / "chroma_db"
)

COLLECTION_NAME = "biesse_knowledge_base"

EXPECTED_DIMENSIONS = 1536


# ============================================================
# ChromaDB client
# ============================================================

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.

    The database is stored locally so the POC does not
    require a cloud vector database.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# ============================================================
# Collection
# ============================================================

def get_collection(
    client: chromadb.PersistentClient,
):
    """
    Get or create the Biesse knowledge-base collection.

    We do NOT configure a Chroma embedding function because
    embeddings have already been generated using Gemini.
    """

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": (
                "Biesse CNC technical "
                "knowledge base"
            ),
            "embedding_model": (
                "gemini-embedding-2"
            ),
            "embedding_dimensions": (
                EXPECTED_DIMENSIONS
            ),
        },
    )

    return collection


# ============================================================
# Load embeddings
# ============================================================

def load_embeddings() -> list[dict[str, Any]]:
    """
    Load embedding records from embeddings.jsonl.

    Expected structure:

    {
        "chunk_id": 0,
        "text": "...",
        "embedding": [...],
        "metadata": {
            "source": "...",
            "start_page": 1,
            "end_page": 23,
            "section": "...",
            "word_count": 877
        }
    }
    """

    if not EMBEDDINGS_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{EMBEDDINGS_FILE}\n\n"
            "Run embeddings.py first."
        )

    records = []

    with EMBEDDINGS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of "
                    f"{EMBEDDINGS_FILE}"
                ) from error

            records.append(record)

    return records


# ============================================================
# Validate embeddings
# ============================================================

def validate_embeddings(
    records: list[dict[str, Any]]
) -> None:
    """
    Validate embedding records before inserting
    them into ChromaDB.
    """

    if not records:

        raise ValueError(
            "No embedding records found."
        )

    seen_ids: set[str] = set()

    for index, record in enumerate(
        records
    ):

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        if "chunk_id" not in record:

            raise ValueError(
                f"Record {index} is missing "
                "'chunk_id'."
            )

        if "text" not in record:

            raise ValueError(
                f"Record {index} is missing "
                "'text'."
            )

        if "embedding" not in record:

            raise ValueError(
                f"Record {index} is missing "
                "'embedding'."
            )

        # ----------------------------------------------------
        # ID validation
        # ----------------------------------------------------

        chunk_id = str(
            record["chunk_id"]
        )

        if chunk_id in seen_ids:

            raise ValueError(
                f"Duplicate chunk_id found: "
                f"{chunk_id}"
            )

        seen_ids.add(
            chunk_id
        )

        # ----------------------------------------------------
        # Embedding validation
        # ----------------------------------------------------

        embedding = record[
            "embedding"
        ]

        if len(embedding) != (
            EXPECTED_DIMENSIONS
        ):

            raise ValueError(
                f"Chunk {chunk_id} has "
                f"{len(embedding)} dimensions. "
                f"Expected "
                f"{EXPECTED_DIMENSIONS}."
            )

        # ----------------------------------------------------
        # Metadata validation
        # ----------------------------------------------------

        metadata = record.get(
            "metadata",
            {}
        )

        if not isinstance(
            metadata,
            dict,
        ):

            raise ValueError(
                f"Chunk {chunk_id} metadata "
                "must be a dictionary."
            )


# ============================================================
# Convert metadata for ChromaDB
# ============================================================

def prepare_metadata(
    record: dict[str, Any]
) -> dict[str, Any]:
    """
    Convert our metadata into Chroma-compatible
    primitive values.

    Chroma metadata values should be strings,
    integers, floats, or booleans.
    """

    metadata = record.get(
        "metadata",
        {}
    )

    result = {
        "source": str(
            metadata.get(
                "source",
                ""
            )
        ),
        "start_page": int(
            metadata.get(
                "start_page",
                0
            )
        ),
        "end_page": int(
            metadata.get(
                "end_page",
                0
            )
        ),
        "section": str(
            metadata.get(
                "section",
                ""
            )
        ),
        "word_count": int(
            metadata.get(
                "word_count",
                0
            )
        ),
    }

    return result


# ============================================================
# Add records to ChromaDB
# ============================================================

def add_embeddings(
    collection,
    records: list[dict[str, Any]],
) -> None:
    """
    Insert embedding records into ChromaDB.

    Uses upsert so running the script multiple times
    does not create duplicate documents.
    """

    ids = [
        str(
            record["chunk_id"]
        )
        for record in records
    ]

    documents = [
        record["text"]
        for record in records
    ]

    embeddings = [
        record["embedding"]
        for record in records
    ]

    metadatas = [
        prepare_metadata(
            record
        )
        for record in records
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ============================================================
# Verify collection
# ============================================================

def verify_collection(
    collection,
    expected_count: int,
) -> None:
    """
    Verify that ChromaDB contains the expected number
    of documents.
    """

    actual_count = (
        collection.count()
    )

    print()
    print("=" * 70)
    print("CHROMADB VERIFICATION")
    print("=" * 70)

    print(
        f"Expected documents: "
        f"{expected_count}"
    )

    print(
        f"Actual documents:   "
        f"{actual_count}"
    )

    if actual_count != expected_count:

        raise RuntimeError(
            f"ChromaDB contains "
            f"{actual_count} documents, "
            f"but expected "
            f"{expected_count}."
        )

    print()
    print(
        "SUCCESS: ChromaDB contains "
        "all expected documents."
    )


# ============================================================
# Preview collection
# ============================================================

def preview_collection(
    collection,
    limit: int = 3,
) -> None:
    """
    Display a few records to verify that the
    data was inserted correctly.
    """

    result = collection.get(
        limit=limit,
        include=[
            "documents",
            "metadatas",
        ],
    )

    print()
    print("=" * 70)
    print("CHROMADB PREVIEW")
    print("=" * 70)

    ids = result.get(
        "ids",
        []
    )

    documents = result.get(
        "documents",
        []
    )

    metadatas = result.get(
        "metadatas",
        []
    )

    for index, chunk_id in enumerate(
        ids
    ):

        print()
        print(
            f"Chunk ID: {chunk_id}"
        )

        if index < len(
            metadatas
        ):

            print(
                f"Source: "
                f"{metadatas[index].get('source')}"
            )

            print(
                f"Pages: "
                f"{metadatas[index].get('start_page')}-"
                f"{metadatas[index].get('end_page')}"
            )

            print(
                f"Section: "
                f"{metadatas[index].get('section')}"
            )

        if index < len(
            documents
        ):

            preview = documents[
                index
            ][:300]

            print(
                f"Text preview: "
                f"{preview}..."
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 70)
    print("BIESSE CHROMADB VECTOR STORE")
    print("=" * 70)

    print(
        f"Embeddings file:\n"
        f"{EMBEDDINGS_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_embeddings()

    print(
        f"Loaded embeddings: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_embeddings(
        records
    )

    print(
        "Embedding validation: PASSED"
    )

    # --------------------------------------------------------
    # Chroma client
    # --------------------------------------------------------

    client = get_chroma_client()

    print(
        f"ChromaDB location:\n"
        f"{CHROMA_DIR}"
    )

    # --------------------------------------------------------
    # Collection
    # --------------------------------------------------------

    collection = get_collection(
        client
    )

    print(
        f"Collection: "
        f"{COLLECTION_NAME}"
    )

    # --------------------------------------------------------
    # Insert
    # --------------------------------------------------------

    add_embeddings(
        collection,
        records,
    )

    print(
        "Embeddings inserted into "
        "ChromaDB."
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_collection(
        collection,
        expected_count=len(records),
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    preview_collection(
        collection,
        limit=3,
    )

    print()
    print("=" * 70)
    print("VECTOR STORE READY")
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()