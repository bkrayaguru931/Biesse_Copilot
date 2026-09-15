import json
import csv
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CONVERSATIONS_DIR = BASE_DIR / "conversations"

DATASET_JSONL = BASE_DIR / "dataset.jsonl"
METADATA_CSV = BASE_DIR / "metadata.csv"


# ============================================================
# Find conversation files
# ============================================================

json_files = sorted(CONVERSATIONS_DIR.glob("*.json"))

if not json_files:
    print("ERROR: No conversation JSON files found.")
    print(f"Expected files inside: {CONVERSATIONS_DIR}")
    exit(1)


print("=" * 60)
print("BIESSE DATASET GENERATION")
print("=" * 60)

print(f"\nFound {len(json_files)} conversation files.")


# ============================================================
# Prepare data
# ============================================================

conversations = []
metadata_rows = []


for file_path in json_files:

    print(f"Processing: {file_path.name}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            conversation = json.load(file)

    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {file_path.name}")
        print(e)
        exit(1)

    conversation_id = conversation["conversation_id"]
    category = conversation["category"]
    customer_tone = conversation["customer_tone"]
    language = conversation["language"]
    duration_target = conversation["duration_target"]
    scenario = conversation["scenario"]
    turns = conversation["turns"]

    # --------------------------------------------------------
    # Store complete conversation for JSONL
    # --------------------------------------------------------

    conversations.append(conversation)

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    total_words = sum(
        len(turn["text"].split())
        for turn in turns
    )

    customer_turns = sum(
        1 for turn in turns
        if turn["speaker"] == "customer"
    )

    agent_turns = sum(
        1 for turn in turns
        if turn["speaker"] == "agent"
    )

    # --------------------------------------------------------
    # Metadata row
    # --------------------------------------------------------

    metadata_rows.append({
        "conversation_id": conversation_id,
        "category": category,
        "customer_tone": customer_tone,
        "language": language,
        "duration_target": duration_target,
        "scenario": scenario,
        "turn_count": len(turns),
        "customer_turns": customer_turns,
        "agent_turns": agent_turns,
        "word_count": total_words
    })


# ============================================================
# Generate dataset.jsonl
# ============================================================

print("\nGenerating dataset.jsonl...")

with open(DATASET_JSONL, "w", encoding="utf-8") as file:

    for conversation in conversations:
        json.dump(
            conversation,
            file,
            ensure_ascii=False
        )
        file.write("\n")


# ============================================================
# Generate metadata.csv
# ============================================================

print("Generating metadata.csv...")

metadata_fields = [
    "conversation_id",
    "category",
    "customer_tone",
    "language",
    "duration_target",
    "scenario",
    "turn_count",
    "customer_turns",
    "agent_turns",
    "word_count"
]

with open(
    METADATA_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=metadata_fields
    )

    writer.writeheader()
    writer.writerows(metadata_rows)


# ============================================================
# Final summary
# ============================================================

total_turns = sum(
    row["turn_count"]
    for row in metadata_rows
)

total_words = sum(
    row["word_count"]
    for row in metadata_rows
)

print()
print("=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)

print(f"\nConversations : {len(conversations)}")
print(f"Total turns   : {total_turns}")
print(f"Total words   : {total_words}")

print(f"\nCreated:")
print(f"  ✓ {DATASET_JSONL.name}")
print(f"  ✓ {METADATA_CSV.name}")

print("\nFiles are ready for validation.")