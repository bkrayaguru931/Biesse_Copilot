import json
from pathlib import Path

from conversation.sentiment import analyze_sentiment
from conversation.category import categorize_query
from rag.search import search


def load_assemblyai_transcript(path: str) -> list:
    """
    Load finalized AssemblyAI turns from the saved transcript.
    """

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your AssemblyAI output stores finalized turns.
    if isinstance(data, dict):

        if "turns" in data:
            return data["turns"]

        if "transcript" in data:
            return data["transcript"]

    if isinstance(data, list):
        return data

    raise ValueError(
        "Could not find conversation turns in AssemblyAI transcript."
    )


def build_conversation_text(turns: list) -> str:
    """
    Convert structured AssemblyAI turns into readable
    conversation text for Gemini and RAG.
    """

    lines = []

    for turn in turns:

        speaker = turn.get(
            "speaker",
            "UNKNOWN"
        )

        text = turn.get(
            "text",
            ""
        ).strip()

        if not text:
            continue

        lines.append(
            f"{speaker}: {text}"
        )

    return "\n".join(lines)


def process_conversation(
    transcript_path: str,
    top_k: int = 3,
) -> dict:
    """
    Run the complete conversation-processing pipeline.

    AssemblyAI transcript
        ↓
    Conversation text
        ↓
    Sentiment
        ↓
    Category
        ↓
    RAG retrieval
    """

    print()
    print("=" * 80)
    print("CONVERSATION PROCESSING")
    print("=" * 80)

    # --------------------------------------------------
    # 1. Load AssemblyAI transcript
    # --------------------------------------------------

    print("\n[1/4] Loading AssemblyAI transcript...")

    turns = load_assemblyai_transcript(
        transcript_path
    )

    print(
        f"Loaded {len(turns)} transcript turns."
    )

    # --------------------------------------------------
    # 2. Build conversation text
    # --------------------------------------------------

    print("\n[2/4] Building conversation context...")

    conversation = build_conversation_text(
        turns
    )

    print(
        f"Conversation length: "
        f"{len(conversation.split())} words"
    )

    # --------------------------------------------------
    # 3. Sentiment
    # --------------------------------------------------

    print("\n[3/4] Analyzing sentiment...")

    sentiment = analyze_sentiment(
        conversation
    )

    print(
        f"Sentiment: "
        f"{sentiment.get('sentiment')}"
    )

    print(
        f"Confidence: "
        f"{sentiment.get('confidence')}"
    )

    # --------------------------------------------------
    # 4. Category
    # --------------------------------------------------

    print("\n[4/4] Categorizing query...")

    category = categorize_query(
        conversation
    )

    print(
        f"Category: "
        f"{category.get('category')}"
    )

    print(
        f"Confidence: "
        f"{category.get('confidence')}"
    )

    # --------------------------------------------------
    # 5. RAG
    # --------------------------------------------------

    print("\n[5/5] Searching knowledge base...")

    relevant_docs = search(
        conversation,
        top_k=top_k,
    )

    print(
        f"Retrieved {len(relevant_docs)} documents."
    )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    result = {
        "transcript": {
            "turn_count": len(turns),
            "conversation": conversation,
        },

        "sentiment": sentiment,

        "category": category,

        "retrieved_documents": relevant_docs,
    }

    return result


def save_result(
    result: dict,
    output_path: str,
):
    """
    Save conversation-processing result.
    """

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Processing result saved to:\n"
        f"{output}"
    )


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            "python conversation\\processor.py "
            "\"path\\to\\transcript.json\""
        )

        sys.exit(1)

    transcript_path = sys.argv[1]

    result = process_conversation(
        transcript_path
    )

    save_result(
        result,
        "biesse_conversation_dataset/processed/"
        "C01_processing.json"
    )