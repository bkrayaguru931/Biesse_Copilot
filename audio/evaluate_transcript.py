import json
import re
from pathlib import Path
from difflib import SequenceMatcher


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "biesse_conversation_dataset"

GT_PATH = DATASET_DIR / "conversations" / "c01.json"
AA_PATH = DATASET_DIR / "transcripts" / "C01_assemblyai.json"

EVALUATION_DIR = DATASET_DIR / "evaluations"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    EVALUATION_DIR / "C01_evaluation.json"
)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """
    Normalize text before comparing transcripts.

    This removes differences caused by:
    - capitalization
    - punctuation
    - extra whitespace
    """

    if not text:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def text_similarity(text1, text2):
    """
    Calculate normalized sequence similarity.
    """

    a = normalize_text(text1)
    b = normalize_text(text2)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# ============================================================
# Find dialogue recursively
# ============================================================

def find_dialogue_list(obj):
    """
    Recursively search JSON for a list containing
    speaker/text dialogue objects.
    """

    if isinstance(obj, list):

        if obj:

            dialogue_like = 0

            for item in obj:

                if isinstance(item, dict):

                    has_text = (
                        "text" in item
                        or "utterance" in item
                    )

                    has_speaker = (
                        "speaker" in item
                        or "role" in item
                    )

                    if has_text and has_speaker:
                        dialogue_like += 1

            if dialogue_like >= 2:
                return obj

        for item in obj:

            result = find_dialogue_list(item)

            if result is not None:
                return result

    elif isinstance(obj, dict):

        for value in obj.values():

            result = find_dialogue_list(value)

            if result is not None:
                return result

    return None


# ============================================================
# Load JSON
# ============================================================

def load_json(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# Extract turns
# ============================================================

def extract_turns(data):

    dialogue = find_dialogue_list(data)

    if dialogue is None:

        raise ValueError(
            "Could not find a dialogue list "
            "containing speaker/text entries."
        )

    turns = []

    for index, item in enumerate(dialogue, start=1):

        if not isinstance(item, dict):
            continue

        text = (
            item.get("text")
            or item.get("utterance")
            or ""
        )

        speaker = (
            item.get("speaker")
            or item.get("role")
            or ""
        )

        if not text.strip():
            continue

        turns.append(
            {
                "index": index,
                "speaker": str(speaker),
                "text": str(text).strip(),
            }
        )

    return turns


# ============================================================
# Speaker normalization
# ============================================================

def normalize_gt_speaker(speaker):

    speaker = speaker.lower().strip()

    if speaker in {
        "customer",
        "caller",
        "user",
        "client",
    }:
        return "customer"

    if speaker in {
        "agent",
        "support_agent",
        "support agent",
        "representative",
        "rep",
    }:
        return "agent"

    return speaker


def normalize_aa_speaker(speaker):

    if speaker is None:
        return None

    speaker = str(speaker).upper().strip()

    if speaker in {
        "",
        "PENDING",
        "UNKNOWN",
        "NONE",
    }:
        return None

    return speaker


# ============================================================
# Combine AssemblyAI turns
# ============================================================

def combine_turns(turns):

    if not turns:
        return {
            "speaker": None,
            "text": "",
            "indices": [],
        }

    speakers = [
        normalize_aa_speaker(
            t["speaker"]
        )
        for t in turns
    ]

    valid_speakers = [
        s for s in speakers
        if s is not None
    ]

    if valid_speakers:

        # Majority speaker
        speaker_counts = {}

        for speaker in valid_speakers:

            speaker_counts[speaker] = (
                speaker_counts.get(
                    speaker,
                    0
                ) + 1
            )

        predicted_speaker = max(
            speaker_counts,
            key=speaker_counts.get
        )

    else:

        predicted_speaker = None

    text = " ".join(
        t["text"]
        for t in turns
    )

    return {
        "speaker": predicted_speaker,
        "text": text.strip(),
        "indices": [
            t["index"]
            for t in turns
        ],
        "raw_speakers": speakers,
    }


# ============================================================
# Dynamic alignment
# ============================================================

def align_turns(gt_turns, aa_turns):

    n = len(gt_turns)
    m = len(aa_turns)

    # --------------------------------------------------------
    # DP matrix
    #
    # Each GT turn may correspond to:
    #   1 AssemblyAI turn
    #   2 AssemblyAI turns
    #   3 AssemblyAI turns
    #   4 AssemblyAI turns
    #
    # This is necessary because streaming ASR can split
    # a natural dialogue turn into multiple events.
    # --------------------------------------------------------

    INF = float("inf")

    dp = [
        [INF] * (m + 1)
        for _ in range(n + 1)
    ]

    back = [
        [None] * (m + 1)
        for _ in range(n + 1)
    ]

    dp[0][0] = 0.0

    MAX_GROUP = 4

    for i in range(n):

        for j in range(m):

            if dp[i][j] == INF:
                continue

            # ------------------------------------------------
            # Try grouping 1-4 AssemblyAI turns together
            # ------------------------------------------------

            for group_size in range(
                1,
                MAX_GROUP + 1
            ):

                end = j + group_size

                if end > m:
                    break

                group = aa_turns[
                    j:end
                ]

                combined = combine_turns(
                    group
                )

                similarity = text_similarity(
                    gt_turns[i]["text"],
                    combined["text"]
                )

                # ------------------------------------------------
                # Cost:
                #
                # High similarity -> low cost
                # ------------------------------------------------

                cost = (
                    1.0 - similarity
                )

                # Small penalty for grouping too many
                # streaming turns.
                cost += (
                    group_size - 1
                ) * 0.01

                new_cost = (
                    dp[i][j]
                    + cost
                )

                if new_cost < dp[i + 1][end]:

                    dp[i + 1][end] = new_cost

                    back[i + 1][end] = (
                        j,
                        group_size
                    )

    # --------------------------------------------------------
    # Recover alignment
    # --------------------------------------------------------

    aligned = []

    i = n
    j = m

    while i > 0:

        previous = back[i][j]

        if previous is None:

            raise RuntimeError(
                "Could not reconstruct "
                "turn alignment."
            )

        previous_j, group_size = previous

        group = aa_turns[
            previous_j:j
        ]

        combined = combine_turns(
            group
        )

        aligned.append(
            {
                "gt_turn": gt_turns[i - 1],
                "aa_turns": group,
                "combined_aa": combined,
            }
        )

        i -= 1
        j = previous_j

    aligned.reverse()

    return aligned


# ============================================================
# Determine A/B → customer/agent mapping
# ============================================================

def determine_speaker_mapping(aligned):

    votes = {
        "A": {
            "customer": 0,
            "agent": 0,
        },
        "B": {
            "customer": 0,
            "agent": 0,
        },
    }

    for item in aligned:

        gt_speaker = normalize_gt_speaker(
            item["gt_turn"]["speaker"]
        )

        aa_speaker = item[
            "combined_aa"
        ]["speaker"]

        if (
            aa_speaker in votes
            and gt_speaker in {
                "customer",
                "agent",
            }
        ):

            votes[
                aa_speaker
            ][
                gt_speaker
            ] += 1

    mapping = {}

    for aa_speaker, counts in votes.items():

        if counts["customer"] == 0 and \
           counts["agent"] == 0:

            continue

        mapping[
            aa_speaker
        ] = max(
            counts,
            key=counts.get
        )

    total_votes = sum(
        sum(v.values())
        for v in votes.values()
    )

    correct_votes = sum(
        max(v.values())
        for v in votes.values()
        if sum(v.values()) > 0
    )

    confidence = (
        correct_votes / total_votes
        if total_votes
        else 0.0
    )

    return (
        mapping,
        votes,
        confidence
    )


# ============================================================
# Evaluate
# ============================================================

def evaluate():

    print()
    print("=" * 80)
    print("ASSEMBLYAI TRANSCRIPT EVALUATION")
    print("=" * 80)

    print()
    print(
        f"Ground truth:\n  {GT_PATH}"
    )

    print(
        f"AssemblyAI:\n  {AA_PATH}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    gt_data = load_json(
        GT_PATH
    )

    aa_data = load_json(
        AA_PATH
    )

    # --------------------------------------------------------
    # Extract turns
    # --------------------------------------------------------

    gt_turns = extract_turns(
        gt_data
    )

    aa_turns = extract_turns(
        aa_data
    )

    print()
    print(
        f"Ground truth turns: {len(gt_turns)}"
    )

    print(
        f"AssemblyAI turns:   {len(aa_turns)}"
    )

    # --------------------------------------------------------
    # Align
    # --------------------------------------------------------

    aligned = align_turns(
        gt_turns,
        aa_turns
    )

    print(
        f"Aligned groups:     {len(aligned)}"
    )

    # --------------------------------------------------------
    # Determine mapping
    # --------------------------------------------------------

    (
        speaker_mapping,
        speaker_votes,
        mapping_confidence,
    ) = determine_speaker_mapping(
        aligned
    )

    print()
    print("Speaker mapping:")

    for aa_speaker, role in speaker_mapping.items():

        print(
            f"  {aa_speaker} -> {role}"
        )

    print(
        f"Mapping confidence: "
        f"{mapping_confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    similarities = []

    speaker_correct = 0
    speaker_evaluable = 0

    pending_turns = 0
    total_aa_turns = len(aa_turns)

    detailed_alignment = []

    for item in aligned:

        gt = item["gt_turn"]
        aa = item["combined_aa"]

        similarity = text_similarity(
            gt["text"],
            aa["text"]
        )

        similarities.append(
            similarity
        )

        gt_role = normalize_gt_speaker(
            gt["speaker"]
        )

        aa_speaker = aa["speaker"]

        predicted_role = None

        if aa_speaker is not None:

            predicted_role = (
                speaker_mapping.get(
                    aa_speaker
                )
            )

        if predicted_role is not None:

            speaker_evaluable += 1

            if predicted_role == gt_role:

                speaker_correct += 1

        # Count PENDING/unknown labels
        for raw_speaker in aa.get(
            "raw_speakers",
            []
        ):

            if raw_speaker is None:
                pending_turns += 1

        detailed_alignment.append(
            {
                "ground_truth_turn": {
                    "index":
                        gt["index"],
                    "speaker":
                        gt["speaker"],
                    "normalized_speaker":
                        gt_role,
                    "text":
                        gt["text"],
                },

                "assemblyai_turns": [
                    {
                        "index":
                            t["index"],
                        "speaker":
                            t["speaker"],
                        "text":
                            t["text"],
                    }
                    for t in item[
                        "aa_turns"
                    ]
                ],

                "combined_assemblyai": {
                    "speaker":
                        aa_speaker,
                    "predicted_role":
                        predicted_role,
                    "text":
                        aa["text"],
                },

                "text_similarity":
                    round(
                        similarity,
                        4
                    ),

                "speaker_correct":
                    (
                        predicted_role
                        == gt_role
                        if predicted_role
                        is not None
                        else None
                    ),
            }
        )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    average_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0.0
    )

    speaker_accuracy = (
        speaker_correct
        / speaker_evaluable
        if speaker_evaluable
        else 0.0
    )

    pending_rate = (
        pending_turns
        / total_aa_turns
        if total_aa_turns
        else 0.0
    )

    alignment_coverage = (
        len(aligned)
        / len(gt_turns)
        if gt_turns
        else 0.0
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"Ground truth turns:       "
        f"{len(gt_turns)}"
    )

    print(
        f"AssemblyAI turns:         "
        f"{len(aa_turns)}"
    )

    print(
        f"Aligned groups:           "
        f"{len(aligned)}"
    )

    print(
        f"Average text similarity:   "
        f"{average_similarity * 100:.2f}%"
    )

    print(
        f"Speaker attribution:       "
        f"{speaker_accuracy * 100:.2f}%"
    )

    print(
        f"Speaker-evaluable groups:  "
        f"{speaker_evaluable}"
    )

    print(
        f"Pending/unknown labels:    "
        f"{pending_turns}"
    )

    print(
        f"Pending label rate:        "
        f"{pending_rate * 100:.2f}%"
    )

    print(
        f"Alignment coverage:        "
        f"{alignment_coverage * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Speaker vote table
    # --------------------------------------------------------

    print()
    print("Speaker mapping votes:")

    for speaker, counts in speaker_votes.items():

        print(
            f"  {speaker}: "
            f"customer={counts['customer']}, "
            f"agent={counts['agent']}"
        )

    # --------------------------------------------------------
    # Detailed mismatches
    # --------------------------------------------------------

    mismatches = [
        item
        for item in detailed_alignment
        if item["speaker_correct"] is False
    ]

    print()
    print(
        f"Speaker mismatches: "
        f"{len(mismatches)}"
    )

    if mismatches:

        print()
        print("First speaker mismatches:")

        for item in mismatches[:10]:

            gt = item[
                "ground_truth_turn"
            ]

            aa = item[
                "combined_assemblyai"
            ]

            print()
            print(
                f"GT #{gt['index']} "
                f"[{gt['normalized_speaker']}]"
            )

            print(
                f"  Expected: "
                f"{gt['text']}"
            )

            print(
                f"  AssemblyAI "
                f"[{aa['speaker']}] "
                f"-> "
                f"{aa['predicted_role']}"
            )

            print(
                f"  {aa['text']}"
            )

    # --------------------------------------------------------
    # Save evaluation
    # --------------------------------------------------------

    evaluation = {

        "conversation_id": "C01",

        "ground_truth_file":
            str(GT_PATH),

        "assemblyai_file":
            str(AA_PATH),

        "metrics": {

            "ground_truth_turns":
                len(gt_turns),

            "assemblyai_turns":
                len(aa_turns),

            "aligned_groups":
                len(aligned),

            "average_text_similarity":
                round(
                    average_similarity,
                    4
                ),

            "speaker_attribution_accuracy":
                round(
                    speaker_accuracy,
                    4
                ),

            "speaker_evaluable_groups":
                speaker_evaluable,

            "pending_or_unknown_turns":
                pending_turns,

            "pending_or_unknown_rate":
                round(
                    pending_rate,
                    4
                ),

            "alignment_coverage":
                round(
                    alignment_coverage,
                    4
                ),
        },

        "speaker_mapping": {

            "mapping":
                speaker_mapping,

            "votes":
                speaker_votes,

            "confidence":
                round(
                    mapping_confidence,
                    4
                ),
        },

        "alignment":
            detailed_alignment,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evaluation,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("EVALUATION SAVED")
    print("=" * 80)

    print(
        OUTPUT_PATH
    )

    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    evaluate()