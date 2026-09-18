import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_PATH = BASE_DIR / "knowledge_base" / "chroma_db"
COLLECTION_NAME = "biesse_knowledge_base"

EMBEDDING_MODEL = "gemini-embedding-2"
OUTPUT_DIMENSIONALITY = 1536

TOP_K = 3


# =============================================================================
# EVALUATION CASES
# =============================================================================
#
# These are representative queries based on the 10 synthetic support calls.
#
# expected_keywords are used as a lightweight relevance signal.
# They are NOT intended to replace manual evaluation.
#

EVALUATION_CASES = [
    {
        "id": "C01",
        "category": "Machine Operation",
        "query": (
            "The machine will not start the automatic machining program. "
            "What should I check?"
        ),
        "expected_keywords": [
            ["automatic", "program", "operation"],
            ["start", "starting"],
        ],
    },
    {
        "id": "C02",
        "category": "Machine Operation",
        "query": (
            "The machining cycle stops unexpectedly during operation "
            "and the machine does not continue the program. What should I check?"
        ),
        "expected_keywords": [
            ["cycle", "program", "operation"],
            ["stop", "stopping", "stops"],
            ["troubleshooting"],
        ],
    },
    {
        "id": "C03",
        "category": "Machine Operation",
        "query": (
            "I am confused about selecting the correct machining program "
            "and preparing the machine for automatic operation. What should I do?"
        ),
        "expected_keywords": [
            ["program", "operation"],
            ["automatic"],
            ["select", "selection"],
        ],
    },
    {
        "id": "C04",
        "category": "Maintenance & Parts",
        "query": (
            "I am performing routine machine maintenance and need to know "
            "which filter should be replaced and how to identify the correct replacement."
        ),
        "expected_keywords": [
            ["maintenance"],
            ["filter"],
        ],
    },
    {
        "id": "C05",
        "category": "Maintenance & Parts",
        "query": (
            "The coolant flow is abnormal during machining and it keeps "
            "interrupting production. What should I check?"
        ),
        "expected_keywords": [
            ["coolant"],
            ["flow"],
        ],
    },
    {
        "id": "C06",
        "category": "Maintenance & Parts",
        "query": (
            "A maintenance component needs to be replaced. "
            "How can I confirm the correct replacement part and documentation?"
        ),
        "expected_keywords": [
            ["maintenance"],
            ["replacement"],
            ["component", "part"],
        ],
    },
    {
        "id": "C07",
        "category": "Technical Troubleshooting",
        "query": (
            "The spindle keeps giving me overload warnings during machining. "
            "What should I check?"
        ),
        "expected_keywords": [
            ["spindle"],
            ["overload", "load"],
            ["troubleshooting"],
        ],
    },
    {
        "id": "C08",
        "category": "Technical Troubleshooting",
        "query": (
            "A machine error that was previously fixed has returned. "
            "What troubleshooting steps should I perform?"
        ),
        "expected_keywords": [
            ["troubleshooting"],
            ["error", "alarm"],
        ],
    },
    {
        "id": "C09",
        "category": "Technical Troubleshooting",
        "query": (
            "A machine communication error appeared after the computer was replaced. "
            "What should I check?"
        ),
        "expected_keywords": [
            ["network", "networking", "communication"],
            ["error"],
        ],
    },
    {
        "id": "C10",
        "category": "Technical Troubleshooting",
        "query": (
            "The machining results became inconsistent after installing a replacement "
            "tool. What should I check?"
        ),
        "expected_keywords": [
            ["tool"],
            ["machining"],
            ["replacement"],
        ],
    },
]


# =============================================================================
# QUERY PREPARATION
# =============================================================================

def prepare_query(query: str) -> str:
    """
    Prepare a user query for Gemini Embedding 2.

    The query is formatted specifically for
    question-answering retrieval.
    """

    query = " ".join(query.split())

    return f"task: question answering | query: {query}"


# =============================================================================
# GEMINI EMBEDDING
# =============================================================================

def create_embedding(client, query: str):
    """
    Generate a single Gemini Embedding 2 vector.

    A separate Content object is used for the input,
    consistent with the current Gemini Embedding 2 API behavior.
    """

    prepared_query = prepare_query(query)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=prepared_query
                    )
                ],
            )
        ],
        config=types.EmbedContentConfig(
            output_dimensionality=OUTPUT_DIMENSIONALITY
        ),
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no embedding.")

    embedding = response.embeddings[0].values

    if len(embedding) != OUTPUT_DIMENSIONALITY:
        raise RuntimeError(
            f"Unexpected embedding dimension: {len(embedding)} "
            f"(expected {OUTPUT_DIMENSIONALITY})"
        )

    return embedding


# =============================================================================
# RELEVANCE CHECKING
# =============================================================================

def get_document_text(result):
    """
    Safely extract text from a ChromaDB result.
    """

    documents = result.get("documents") or [[]]

    if not documents[0]:
        return ""

    return documents[0][0] or ""


def get_metadata(result, index):
    """
    Safely extract metadata for one result.
    """

    metadatas = result.get("metadatas") or [[]]

    if len(metadatas[0]) <= index:
        return {}

    return metadatas[0][index] or {}


def contains_any(text: str, words):
    """
    Return True if at least one word is present.
    """

    text = text.lower()

    return any(
        word.lower() in text
        for word in words
    )


def evaluate_result(metadata, document, expected_keywords):
    """
    Lightweight relevance heuristic.

    Each keyword group represents an expected concept.
    A result passes if at least one word from every group
    appears in its metadata/content.

    Example:

        [
            ["spindle"],
            ["overload", "load"],
            ["troubleshooting"]
        ]

    means all three concepts should be represented.
    """

    searchable_text = " ".join(
        [
            str(metadata.get("section", "")),
            str(metadata.get("source", "")),
            document,
        ]
    ).lower()

    matched_groups = 0

    for group in expected_keywords:
        if contains_any(searchable_text, group):
            matched_groups += 1

    return matched_groups == len(expected_keywords), matched_groups


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Load environment
    # -------------------------------------------------------------------------

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the environment."
        )

    # -------------------------------------------------------------------------
    # Initialize clients
    # -------------------------------------------------------------------------

    client = genai.Client(
        api_key=api_key
    )

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

    collection_count = collection.count()

    print(f"ChromaDB path : {CHROMA_PATH}")
    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Documents     : {collection_count}")
    print(f"Embedding     : {EMBEDDING_MODEL}")
    print(f"Dimensions    : {OUTPUT_DIMENSIONALITY}")
    print(f"Top-K         : {TOP_K}")
    print()

    # -------------------------------------------------------------------------
    # Evaluation counters
    # -------------------------------------------------------------------------

    top1_correct = 0
    top3_correct = 0

    detailed_results = []

    # -------------------------------------------------------------------------F
    # Run evaluationrerank
    F
    # -------------------------------------------------------------------------

    for case in EVALUATION_CASES:

        print("-" * 80)
        print(f"{case['id']} | {case['category']}")
        print("-" * 80)

        print(f"Query:")
        print(case["query"])
        print()

        try:
            query_embedding = create_embedding(
                client,
                case["query"]
            )

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

        except Exception as exc:
            print(f"ERROR: {exc}")
            print()

            detailed_results.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "error": str(exc),
                }
            )

            continue

        case_top1 = False
        case_top3 = False

        retrieved = []

        for rank in range(TOP_K):

            metadata = get_metadata(
                results,
                rank
            )

            document = get_document_text(
                {
                    "documents": [
                        [results["documents"][0][rank]]
                    ]
                }
            )

            distance = results["distances"][0][rank]

            relevant, matched_groups = evaluate_result(
                metadata,
                document,
                case["expected_keywords"]
            )

            if rank == 0 and relevant:
                case_top1 = True

            if relevant:
                case_top3 = True

            source = metadata.get(
                "source",
                "Unknown"
            )

            section = metadata.get(
                "section",
                "Unknown"
            )

            start_page = metadata.get(
                "start_page",
                "?"
            )

            end_page = metadata.get(
                "end_page",
                "?"
            )

            print(
                f"Rank {rank + 1}: "
                f"distance={distance:.4f}"
            )

            print(
                f"  Source : {source}"
            )

            print(
                f"  Pages  : {start_page}-{end_page}"
            )

            print(
                f"  Section: {section}"
            )

            print(
                f"  Match  : "
                f"{'RELEVANT' if relevant else 'not matched'} "
                f"({matched_groups}/{len(case['expected_keywords'])} "
                f"concept groups)"
            )

            preview = " ".join(
                document.split()
            )

            if len(preview) > 250:
                preview = preview[:250] + "..."

            print(
                f"  Preview: {preview}"
            )

            print()

            retrieved.append(
                {
                    "rank": rank + 1,
                    "distance": distance,
                    "source": source,
                    "start_page": start_page,
                    "end_page": end_page,
                    "section": section,
                    "relevant": relevant,
                    "matched_groups": matched_groups,
                }
            )

        if case_top1:
            top1_correct += 1

        if case_top3:
            top3_correct += 1

        detailed_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "query": case["query"],
                "top1_relevant": case_top1,
                "top3_relevant": case_top3,
                "results": retrieved,
            }
        )

        print(
            f"Evaluation: "
            f"Top-1={'PASS' if case_top1 else 'FAIL'} | "
            f"Top-3={'PASS' if case_top3 else 'FAIL'}"
        )

        print()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    total_cases = len(EVALUATION_CASES)

    top1_accuracy = (
        top1_correct / total_cases
        if total_cases
        else 0
    )

    top3_recall = (
        top3_correct / total_cases
        if total_cases
        else 0
    )

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print()

    print(
        f"Total test cases       : {total_cases}"
    )

    print(
        f"Top-1 relevant        : "
        f"{top1_correct}/{total_cases}"
    )

    print(
        f"Top-1 accuracy        : "
        f"{top1_accuracy:.1%}"
    )

    print(
        f"Top-3 relevant        : "
        f"{top3_correct}/{total_cases}"
    )

    print(
        f"Top-3 recall          : "
        f"{top3_recall:.1%}"
    )

    print()

    # -------------------------------------------------------------------------
    # Case summary
    # -------------------------------------------------------------------------

    print("-" * 80)
    print("CASE SUMMARY")
    print("-" * 80)

    for result in detailed_results:

        if "error" in result:
            status = "ERROR"

        else:
            top1 = (
                "PASS"
                if result["top1_relevant"]
                else "FAIL"
            )

            top3 = (
                "PASS"
                if result["top3_relevant"]
                else "FAIL"
            )

            status = f"Top-1 {top1} | Top-3 {top3}"

        print(
            f"{result['id']}: {status}"
        )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    output_path = (
        BASE_DIR
        / "knowledge_base"
        / "processed"
        / "retrieval_evaluation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluation_output = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": OUTPUT_DIMENSIONALITY,
        "top_k": TOP_K,
        "total_cases": total_cases,
        "top1_relevant": top1_correct,
        "top1_accuracy": top1_accuracy,
        "top3_relevant": top3_correct,
        "top3_recall": top3_recall,
        "cases": detailed_results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            evaluation_output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Detailed results saved to:"
    )
    print(
        f"{output_path}"
    )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()