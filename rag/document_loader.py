from pathlib import Path
from pypdf import PdfReader


MANUALS_DIR = Path("knowledge_base/manuals")


def load_pdf(pdf_path: Path):
    """Extract text page-by-page from a PDF."""

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                },
            }
        )

    return documents


def load_all_manuals():
    """Load all PDF manuals from the knowledge-base directory."""

    all_documents = []

    for pdf_path in MANUALS_DIR.glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")

        documents = load_pdf(pdf_path)

        print(f"  Pages extracted: {len(documents)}")

        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":
    documents = load_all_manuals()

    print()
    print("=" * 60)
    print(f"Total extracted pages: {len(documents)}")
    print("=" * 60)

    if documents:
        print("\nExample document:")
        print("Source:", documents[0]["metadata"]["source"])
        print("Page:", documents[0]["metadata"]["page"])
        print("\nText preview:")
        print(documents[0]["text"][:1000])