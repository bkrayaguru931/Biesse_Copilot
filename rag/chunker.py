"""
Production-style document chunker for the Biesse RAG POC.

Pipeline:
    PDF
      ↓
    page extraction
      ↓
    page classification
      ↓
    text normalization
      ↓
    section detection
      ↓
    semantic-ish chunking
      ↓
    overlap
      ↓
    metadata-rich chunks

Design goals:
- Preserve technical sections where possible.
- Avoid embedding obvious TOC/front-matter pages.
- Target approximately 1,000 words per chunk.
- Keep a soft maximum of 1,300 words.
- Maintain ~150 words of overlap.
- Preserve source, page range, and section provenance.
"""

import re

import json
from pathlib import Path
from document_loader import load_all_manuals


# ============================================================================
# Configuration
# ============================================================================

TARGET_WORDS = 1000
SOFT_MAX_WORDS = 1300
OVERLAP_WORDS = 150

MIN_PAGE_WORDS = 40
MIN_CHUNK_WORDS = 250


# ============================================================================
# Text utilities
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize PDF-extracted text without destroying useful structure.
    """

    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces but preserve line breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_words(text: str) -> list[str]:
    """Return whitespace-separated words."""
    return text.split()


# ============================================================================
# Page classification
# ============================================================================

def looks_like_table_of_contents(text: str) -> bool:
    """
    Detect pages that are primarily table-of-contents material.

    Uses multiple signals instead of relying on page numbers.
    """

    if not text:
        return False

    lower = text.lower()

    strong_keywords = [
        "table of contents",
        "contents",
        "list of figures",
        "list of tables",
    ]

    keyword_hits = sum(
        1 for keyword in strong_keywords
        if keyword in lower
    )

    # PDF TOCs frequently contain dotted leaders:
    #
    # Setting up the machine ................. 42
    #
    dotted_leaders = len(
        re.findall(r"\.{3,}", text)
    )

    # Section numbering such as:
    # 1.1
    # 3.4.2
    # 7.12.1.3
    section_numbers = len(
        re.findall(r"\b\d+(?:\.\d+){1,4}\b", text)
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    line_count = max(len(lines), 1)

    dotted_ratio = dotted_leaders / line_count

    if keyword_hits > 0:
        return True

    if dotted_ratio >= 0.20 and section_numbers >= 8:
        return True

    return False


def classify_page(text: str) -> str:
    """
    Classify a PDF page as:

        empty
        front_matter
        toc
        content
    """

    text = normalize_text(text)

    if not text:
        return "empty"

    word_count = len(get_words(text))

    if word_count < MIN_PAGE_WORDS:
        return "front_matter"

    if looks_like_table_of_contents(text):
        return "toc"

    return "content"


# ============================================================================
# Heading detection
# ============================================================================

def is_numbered_heading(line: str) -> bool:
    """
    Detect headings such as:

        5 Setting up the machine
        5.4 Setting the zero offset
        7.12.1 General procedure
    """

    pattern = (
        r"^\d+(?:\.\d+){0,4}"
        r"\s+"
        r"[A-Z][A-Za-z0-9 /&(),:'\-_]+$"
    )

    return bool(re.match(pattern, line.strip()))


def is_short_heading(line: str) -> bool:
    """
    Detect short technical headings.

    This intentionally remains conservative because PDF extraction
    can make ordinary sentences look like headings.
    """

    line = line.strip()

    if not line:
        return False

    if len(line) > 120:
        return False

    if len(line.split()) > 12:
        return False

    technical_keywords = [
        "troubleshooting",
        "maintenance",
        "safety instructions",
        "alarm",
        "error",
        "system messages",
        "tool management",
        "spindle",
        "coolant",
        "networking",
        "installation",
        "operation",
        "setting up",
        "diagnostics",
    ]

    lower = line.lower()

    return any(
        keyword in lower
        for keyword in technical_keywords
    )


def looks_like_heading(line: str) -> bool:
    """Return True when a line looks like a section heading."""

    return (
        is_numbered_heading(line)
        or is_short_heading(line)
    )


# ============================================================================
# Section extraction
# ============================================================================

def extract_sections(page: dict) -> list[dict]:
    """
    Convert a content page into logical sections.

    Each returned section retains its original PDF page number.
    """

    text = normalize_text(page["text"])

    if not text:
        return []

    source = page["metadata"]["source"]
    page_number = page["metadata"]["page"]

    lines = text.splitlines()

    sections = []

    current_lines = []
    current_heading = None

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if looks_like_heading(line):

            if current_lines:

                section_text = "\n".join(
                    current_lines
                ).strip()

                if section_text:
                    sections.append(
                        {
                            "text": section_text,
                            "source": source,
                            "start_page": page_number,
                            "end_page": page_number,
                            "section": current_heading,
                        }
                    )

            current_heading = line
            current_lines = [line]

        else:
            current_lines.append(line)

    if current_lines:

        section_text = "\n".join(
            current_lines
        ).strip()

        if section_text:
            sections.append(
                {
                    "text": section_text,
                    "source": source,
                    "start_page": page_number,
                    "end_page": page_number,
                    "section": current_heading,
                }
            )

    return sections


# ============================================================================
# Chunk creation
# ============================================================================

def create_chunk(
    chunk_words: list[str],
    source: str,
    start_page: int,
    end_page: int,
    section: str | None,
) -> dict | None:
    """
    Build a metadata-rich chunk.
    """

    if not chunk_words:
        return None

    text = " ".join(chunk_words).strip()

    if not text:
        return None

    return {
        "text": text,
        "metadata": {
            "source": source,
            "start_page": start_page,
            "end_page": end_page,
            "section": section or "Unknown",
            "word_count": len(chunk_words),
        },
    }


def split_large_section(
    section: dict,
    existing_overlap: list[str],
) -> tuple[list[dict], list[str]]:
    """
    Split an unusually large section into controlled chunks.

    This is only used when a single logical section is itself
    larger than the normal chunk size.
    """

    section_words = get_words(section["text"])

    chunks = []

    current = existing_overlap.copy()

    source = section["source"]
    start_page = section["start_page"]
    end_page = section["end_page"]
    section_name = section["section"]

    while section_words:

        available = SOFT_MAX_WORDS - len(current)

        if available <= 0:

            chunk = create_chunk(
                current,
                source,
                start_page,
                end_page,
                section_name,
            )

            if chunk:
                chunks.append(chunk)

            current = current[-OVERLAP_WORDS:]

            continue

        take = min(
            available,
            len(section_words),
        )

        current.extend(
            section_words[:take]
        )

        section_words = section_words[take:]

        if len(current) >= TARGET_WORDS:

            chunk = create_chunk(
                current,
                source,
                start_page,
                end_page,
                section_name,
            )

            if chunk:
                chunks.append(chunk)

            current = current[-OVERLAP_WORDS:]

    return chunks, current


def build_chunks(sections: list[dict]) -> list[dict]:
    """
    Combine logical sections into RAG chunks.

    Strategy:
    1. Keep sections together when possible.
    2. Target ~1,000 words.
    3. Use 1,300 as a soft maximum.
    4. Use ~150 words of overlap.
    5. Never mix different source PDFs.
    """

    chunks = []

    current_words = []
    current_source = None
    current_start_page = None
    current_end_page = None
    current_section = None

    def flush():

        nonlocal current_words
        nonlocal current_source
        nonlocal current_start_page
        nonlocal current_end_page
        nonlocal current_section

        if not current_words:
            return

        chunk = create_chunk(
            current_words,
            current_source,
            current_start_page,
            current_end_page,
            current_section,
        )

        if chunk:
            chunks.append(chunk)

    for section in sections:

        section_words = get_words(
            section["text"]
        )

        if not section_words:
            continue

        source = section["source"]
        start_page = section["start_page"]
        end_page = section["end_page"]
        section_name = section["section"]

        # ---------------------------------------------------------------
        # Never mix different manuals in the same chunk.
        # ---------------------------------------------------------------

        if (
            current_source is not None
            and source != current_source
        ):

            flush()

            current_words = []
            current_source = None
            current_start_page = None
            current_end_page = None
            current_section = None

        if current_source is None:

            current_source = source
            current_start_page = start_page
            current_end_page = end_page
            current_section = section_name

        proposed_size = (
            len(current_words)
            + len(section_words)
        )

        # ---------------------------------------------------------------
        # Section fits naturally into current chunk.
        # ---------------------------------------------------------------

        if proposed_size <= TARGET_WORDS:

            current_words.extend(
                section_words
            )

            current_end_page = end_page

            if section_name:
                current_section = section_name

            continue

        # ---------------------------------------------------------------
        # Current chunk is already healthy.
        # Flush it before starting the next section.
        # ---------------------------------------------------------------

        if len(current_words) >= MIN_CHUNK_WORDS:

            flush()

            overlap = current_words[
                -OVERLAP_WORDS:
            ]

            current_words = overlap.copy()

            current_start_page = start_page
            current_end_page = end_page
            current_section = section_name

        # ---------------------------------------------------------------
        # If adding the section still fits under the soft maximum,
        # keep the section intact.
        # ---------------------------------------------------------------

        if (
            len(current_words)
            + len(section_words)
            <= SOFT_MAX_WORDS
        ):

            current_words.extend(
                section_words
            )

            current_end_page = end_page

            if section_name:
                current_section = section_name

            continue

        # ---------------------------------------------------------------
        # Large section: split only because it cannot fit naturally.
        # ---------------------------------------------------------------

        flush()

        overlap = current_words[
            -OVERLAP_WORDS:
        ]

        current_words = overlap.copy()

        current_start_page = start_page
        current_end_page = end_page
        current_section = section_name

        large_chunks, remainder = split_large_section(
            section,
            current_words,
        )

        chunks.extend(large_chunks)

        current_words = remainder.copy()

        current_source = source
        current_start_page = start_page
        current_end_page = end_page
        current_section = section_name

    # ------------------------------------------------------------------
    # Final chunk.
    # ------------------------------------------------------------------

    flush()

    # ------------------------------------------------------------------
    # Stable chunk IDs.
    # ------------------------------------------------------------------

    for chunk_id, chunk in enumerate(chunks):

        chunk["metadata"]["chunk_id"] = chunk_id

    return chunks


# ============================================================================
# Main pipeline
# ============================================================================

def create_chunks() -> list[dict]:

    pages = load_all_manuals()

    print()
    print(f"Total pages loaded: {len(pages)}")

    page_type_counts = {
        "content": 0,
        "toc": 0,
        "front_matter": 0,
        "empty": 0,
    }

    sections = []

    for page in pages:

        page_type = classify_page(
            page["text"]
        )

        page_type_counts[page_type] += 1

        # Only technical content enters the RAG corpus.
        if page_type != "content":
            continue

        page_sections = extract_sections(
            page
        )

        sections.extend(
            page_sections
        )

    print()
    print("Page classification:")
    print(
        f"  Content:      "
        f"{page_type_counts['content']}"
    )
    print(
        f"  TOC:          "
        f"{page_type_counts['toc']}"
    )
    print(
        f"  Front matter: "
        f"{page_type_counts['front_matter']}"
    )
    print(
        f"  Empty:        "
        f"{page_type_counts['empty']}"
    )

    print()
    print(
        f"Logical sections: {len(sections)}"
    )

    chunks = build_chunks(
        sections
    )

    return chunks


# ============================================================================
# Validation
# ============================================================================

def print_statistics(chunks: list[dict]):

    if not chunks:
        print("No chunks generated.")
        return

    word_counts = [
        chunk["metadata"]["word_count"]
        for chunk in chunks
    ]

    average_words = (
        sum(word_counts)
        / len(word_counts)
    )

    print()
    print("=" * 60)
    print("Chunking Results")
    print("=" * 60)

    print(
        f"Total chunks:       {len(chunks)}"
    )

    print(
        f"Average words:      "
        f"{average_words:.0f}"
    )

    print(
        f"Smallest chunk:     "
        f"{min(word_counts)}"
    )

    print(
        f"Largest chunk:      "
        f"{max(word_counts)}"
    )

    # Useful quality checks.
    over_target = sum(
        1
        for count in word_counts
        if count > TARGET_WORDS
    )

    over_soft_max = sum(
        1
        for count in word_counts
        if count > SOFT_MAX_WORDS
    )

    very_small = sum(
        1
        for count in word_counts
        if count < MIN_CHUNK_WORDS
    )

    print()
    print("Quality checks:")
    print(
        f"  Chunks > target ({TARGET_WORDS}): "
        f"{over_target}"
    )

    print(
        f"  Chunks > soft max ({SOFT_MAX_WORDS}): "
        f"{over_soft_max}"
    )

    print(
        f"  Chunks < minimum ({MIN_CHUNK_WORDS}): "
        f"{very_small}"
    )


# ============================================================================
# Sample output
# ============================================================================

def print_samples(chunks: list[dict], count: int = 5):

    print()
    print("=" * 60)
    print("Sample Chunks")
    print("=" * 60)

    for index, chunk in enumerate(
        chunks[:count],
        start=1,
    ):

        metadata = chunk["metadata"]

        print()
        print(
            f"CHUNK {index}"
        )

        print("-" * 60)

        print(
            f"ID:          "
            f"{metadata['chunk_id']}"
        )

        print(
            f"Source:      "
            f"{metadata['source']}"
        )

        print(
            f"Pages:       "
            f"{metadata['start_page']} - "
            f"{metadata['end_page']}"
        )

        print(
            f"Section:     "
            f"{metadata['section']}"
        )

        print(
            f"Word count:  "
            f"{metadata['word_count']}"
        )

        print()

        print(
            chunk["text"][:800]
        )

        print("...")


# ============================================================================
# Entry point
# ============================================================================

def main():

    print("=" * 60)
    print("RAG Chunking Pipeline")
    print("=" * 60)

    chunks = create_chunks()

    print_statistics(chunks)

    print_samples(chunks)

    save_chunks(chunks)


def save_chunks(chunks):
    output_dir = (
        Path(__file__).resolve().parent.parent
        / "knowledge_base"
        / "processed"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "chunks.jsonl"

    with output_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(chunk, ensure_ascii=False)
                + "\n"
            )

    print(f"\nSaved {len(chunks)} chunks to:")
    print(output_file)


if __name__ == "__main__":
    main()


