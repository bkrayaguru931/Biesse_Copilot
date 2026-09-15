import json
from pathlib import Path
from collections import Counter


# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path(__file__).resolve().parent
CONVERSATIONS_DIR = DATASET_DIR / "conversations"

EXPECTED_CONVERSATIONS = 10

VALID_CATEGORIES = {
    "Machine Operation",
    "Maintenance & Parts",
    "Technical Troubleshooting"
}

VALID_TONES = {
    "Calm",
    "Frustrated",
    "Confused",
    "Neutral",
    "Agitated",
    "Angry"
}

VALID_SPEAKERS = {
    "customer",
    "agent"
}

REQUIRED_TOP_LEVEL_FIELDS = {
    "conversation_id",
    "category",
    "customer_tone",
    "language",
    "duration_target",
    "scenario",
    "turns"
}

REQUIRED_TURN_FIELDS = {
    "turn_id",
    "speaker",
    "text"
}


# ============================================================
# Helper functions
# ============================================================

def estimate_duration(word_count):
    """
    Estimate spoken duration assuming approximately
    130 words per minute for natural support conversation.
    """
    words_per_minute = 130
    return word_count / words_per_minute


def format_duration(minutes):
    """Format duration as minutes and seconds."""
    total_seconds = round(minutes * 60)
    mins = total_seconds // 60
    secs = total_seconds % 60

    return f"{mins}m {secs:02d}s"


def print_error(errors, message):
    errors.append(message)


# ============================================================
# Validate one conversation
# ============================================================

def validate_conversation(file_path):
    errors = []
    warnings = []

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return errors, warnings, None

    except Exception as e:
        errors.append(f"Could not read file: {e}")
        return errors, warnings, None

    # --------------------------------------------------------
    # Check top-level structure
    # --------------------------------------------------------

    if not isinstance(data, dict):
        errors.append("Root JSON must be an object.")
        return errors, warnings, None

    missing_fields = REQUIRED_TOP_LEVEL_FIELDS - data.keys()

    if missing_fields:
        errors.append(
            f"Missing top-level fields: {sorted(missing_fields)}"
        )

    # --------------------------------------------------------
    # Conversation ID
    # --------------------------------------------------------

    conversation_id = data.get("conversation_id")

    if not isinstance(conversation_id, str) or not conversation_id.strip():
        errors.append("conversation_id must be a non-empty string.")

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    category = data.get("category")

    if category not in VALID_CATEGORIES:
        errors.append(
            f"Invalid category: {category!r}. "
            f"Expected one of: {sorted(VALID_CATEGORIES)}"
        )

    # --------------------------------------------------------
    # Customer tone
    # --------------------------------------------------------

    tone = data.get("customer_tone")

    if tone not in VALID_TONES:
        errors.append(
            f"Invalid customer_tone: {tone!r}. "
            f"Expected one of: {sorted(VALID_TONES)}"
        )

    # --------------------------------------------------------
    # Language
    # --------------------------------------------------------

    language = data.get("language")

    if not isinstance(language, str) or not language.strip():
        errors.append("language must be a non-empty string.")

    # --------------------------------------------------------
    # Duration target
    # --------------------------------------------------------

    duration_target = data.get("duration_target")

    if not isinstance(duration_target, str):
        errors.append("duration_target must be a string.")

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    scenario = data.get("scenario")

    if not isinstance(scenario, str) or not scenario.strip():
        errors.append("scenario must be a non-empty string.")

    # --------------------------------------------------------
    # Turns
    # --------------------------------------------------------

    turns = data.get("turns")

    if not isinstance(turns, list):
        errors.append("turns must be a list.")
        return errors, warnings, data

    if len(turns) == 0:
        errors.append("Conversation contains no turns.")
        return errors, warnings, data

    if len(turns) < 20:
        warnings.append(
            f"Only {len(turns)} turns. "
            "This may be too short for a 2-5 minute conversation."
        )

    # --------------------------------------------------------
    # Validate individual turns
    # --------------------------------------------------------

    expected_turn_id = 1
    speakers = []
    words = []

    for index, turn in enumerate(turns, start=1):

        if not isinstance(turn, dict):
            errors.append(
                f"Turn {index}: turn must be a JSON object."
            )
            continue

        missing_turn_fields = REQUIRED_TURN_FIELDS - turn.keys()

        if missing_turn_fields:
            errors.append(
                f"Turn {index}: missing fields "
                f"{sorted(missing_turn_fields)}"
            )

        # ----------------------------------------------------
        # turn_id
        # ----------------------------------------------------

        turn_id = turn.get("turn_id")

        if turn_id != expected_turn_id:
            errors.append(
                f"Turn {index}: expected turn_id "
                f"{expected_turn_id}, got {turn_id!r}"
            )

        expected_turn_id += 1

        # ----------------------------------------------------
        # speaker
        # ----------------------------------------------------

        speaker = turn.get("speaker")

        if speaker not in VALID_SPEAKERS:
            errors.append(
                f"Turn {index}: invalid speaker {speaker!r}. "
                f"Expected 'customer' or 'agent'."
            )
        else:
            speakers.append(speaker)

        # ----------------------------------------------------
        # text
        # ----------------------------------------------------

        text = turn.get("text")

        if not isinstance(text, str):
            errors.append(
                f"Turn {index}: text must be a string."
            )

        elif not text.strip():
            errors.append(
                f"Turn {index}: text cannot be empty."
            )

        else:
            words.extend(text.split())

    # --------------------------------------------------------
    # Speaker analysis
    # --------------------------------------------------------

    speaker_counts = Counter(speakers)

    customer_turns = speaker_counts.get("customer", 0)
    agent_turns = speaker_counts.get("agent", 0)

    if customer_turns == 0:
        errors.append("No customer turns found.")

    if agent_turns == 0:
        errors.append("No agent turns found.")

    # --------------------------------------------------------
    # Check consecutive same-speaker turns
    # --------------------------------------------------------

    for i in range(1, len(speakers)):
        if speakers[i] == speakers[i - 1]:
            warnings.append(
                f"Turns {i} and {i + 1} are both "
                f"from '{speakers[i]}'."
            )

    # --------------------------------------------------------
    # Word count / duration
    # --------------------------------------------------------

    word_count = len(words)
    estimated_minutes = estimate_duration(word_count)

    if word_count < 250:
        warnings.append(
            f"Only {word_count} words. "
            "This is likely shorter than 2 minutes."
        )

    if estimated_minutes > 5.5:
        warnings.append(
            f"Estimated duration is {format_duration(estimated_minutes)}, "
            "which may exceed the 5-minute target."
        )

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    stats = {
        "conversation_id": conversation_id,
        "category": category,
        "tone": tone,
        "turns": len(turns),
        "customer_turns": customer_turns,
        "agent_turns": agent_turns,
        "words": word_count,
        "estimated_minutes": estimated_minutes
    }

    return errors, warnings, stats


# ============================================================
# Main validation
# ============================================================

def main():

    print("=" * 65)
    print("BIESSE CONVERSATION DATASET VALIDATION")
    print("=" * 65)

    if not CONVERSATIONS_DIR.exists():
        print()
        print("ERROR: conversations folder not found.")
        print(f"Expected: {CONVERSATIONS_DIR}")
        return

    json_files = sorted(CONVERSATIONS_DIR.glob("*.json"))

    print()
    print(f"Dataset directory : {DATASET_DIR}")
    print(f"Conversations     : {CONVERSATIONS_DIR}")
    print(f"JSON files found  : {len(json_files)}")
    print()

    # --------------------------------------------------------
    # Check number of conversations
    # --------------------------------------------------------

    if len(json_files) != EXPECTED_CONVERSATIONS:
        print(
            f"WARNING: Expected {EXPECTED_CONVERSATIONS} "
            f"JSON files but found {len(json_files)}."
        )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    all_stats = []
    total_errors = 0
    total_warnings = 0

    for file_path in json_files:

        errors, warnings, stats = validate_conversation(file_path)

        print("-" * 65)

        if errors:
            print(f"✗ {file_path.name} - INVALID")

            for error in errors:
                print(f"    ERROR: {error}")

            total_errors += len(errors)

        else:
            print(f"✓ {file_path.name} - VALID")

        if warnings:
            for warning in warnings:
                print(f"    WARNING: {warning}")

            total_warnings += len(warnings)

        if stats:
            all_stats.append(stats)

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)

    total_turns = sum(s["turns"] for s in all_stats)
    total_customer_turns = sum(
        s["customer_turns"] for s in all_stats
    )
    total_agent_turns = sum(
        s["agent_turns"] for s in all_stats
    )
    total_words = sum(s["words"] for s in all_stats)

    print()
    print(f"Total conversations : {len(all_stats)}")
    print(f"Total turns         : {total_turns}")
    print(f"Customer turns      : {total_customer_turns}")
    print(f"Agent turns         : {total_agent_turns}")
    print(f"Total words         : {total_words}")

    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    print()
    print("Categories:")

    category_counts = Counter(
        s["category"] for s in all_stats
    )

    for category in sorted(VALID_CATEGORIES):
        print(
            f"  {category:<30}: "
            f"{category_counts.get(category, 0)}"
        )

    # --------------------------------------------------------
    # Tone distribution
    # --------------------------------------------------------

    print()
    print("Customer tones:")

    tone_counts = Counter(
        s["tone"] for s in all_stats
    )

    for tone in sorted(VALID_TONES):
        print(
            f"  {tone:<20}: "
            f"{tone_counts.get(tone, 0)}"
        )

    # --------------------------------------------------------
    # Conversation statistics
    # --------------------------------------------------------

    print()
    print("-" * 65)
    print("CONVERSATION DETAILS")
    print("-" * 65)

    for stats in all_stats:

        print(
            f"{stats['conversation_id']} | "
            f"{stats['category']:<28} | "
            f"{stats['tone']:<10} | "
            f"{stats['turns']:>2} turns | "
            f"{stats['words']:>3} words | "
            f"~{format_duration(stats['estimated_minutes'])}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 65)

    if total_errors == 0:
        print("✓ DATASET STRUCTURE IS VALID")

        if total_warnings > 0:
            print(
                f"⚠ {total_warnings} warning(s) found. "
                "Review them before generating audio."
            )
        else:
            print("✓ NO WARNINGS")
    else:
        print(
            f"✗ DATASET HAS {total_errors} ERROR(S)"
        )
        print(
            "Fix the errors before proceeding to audio generation."
        )

    print("=" * 65)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()