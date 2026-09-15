import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "gemini-3.6-flash"


# =============================================================================
# PROMPT
# =============================================================================

SYSTEM_INSTRUCTION = """
You are an AI assistant supporting a technical customer-support agent
for industrial CNC machinery.

Your job is to generate a concise, practical troubleshooting suggestion
for the support agent.

You will receive:

1. The current customer-support conversation.
2. Relevant technical documentation retrieved from the local knowledge base.

IMPORTANT RULES:

- Use the retrieved documentation as the primary technical source.
- Do not invent machine-specific facts, procedures, alarm codes, parameters,
  part numbers, or troubleshooting steps that are not supported by the
  retrieved documentation.
- If the documentation does not contain enough information to determine
  the solution, explicitly say that more information is required.
- Do not pretend that generic documentation is Biesse-specific.
- Do not tell the support agent to perform unsafe actions.
- Prefer simple diagnostic questions and checks before recommending
  component replacement.
- The output is a suggestion for the support agent, not a direct final
  response to the customer.
- Keep the response concise and actionable.

Return the following structure:

SUMMARY:
A one or two sentence summary of the likely issue based only on the
conversation and retrieved documentation.

SUGGESTED ACTIONS:
A numbered list of practical troubleshooting checks supported by the
documentation.

QUESTIONS TO ASK:
A short list of additional information the support agent should obtain
from the customer if needed.

SOURCES:
List the document name and page range used for the suggestion.

If the retrieved documentation does not sufficiently address the issue,
say so clearly instead of guessing.
"""


# =============================================================================
# GEMINI CLIENT
# =============================================================================

def create_client():
    """
    Create the Gemini API client.
    """

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in the environment."
        )

    return genai.Client(
        api_key=api_key
    )


# =============================================================================
# DOCUMENT FORMATTING
# =============================================================================

def format_documents(documents):
    """
    Convert retrieved RAG documents into a readable context
    for Gemini.
    """

    formatted = []

    for index, document in enumerate(documents, start=1):

        metadata = document.get("metadata", {})

        source = metadata.get(
            "source",
            "Unknown source"
        )

        start_page = metadata.get(
            "start_page",
            "?"
        )

        end_page = metadata.get(
            "end_page",
            "?"
        )

        section = metadata.get(
            "section",
            "Unknown section"
        )

        text = document.get(
            "text",
            ""
        )

        formatted.append(
            f"""
DOCUMENT {index}

Source:
{source}

Pages:
{start_page}-{end_page}

Section:
{section}

Content:
{text}
"""
        )

    return "\n".join(formatted)


# =============================================================================
# SUGGESTION GENERATION
# =============================================================================

def generate_suggestion(
    conversation,
    documents
):
    """
    Generate a support-agent suggestion using the conversation
    and retrieved technical documentation.
    """

    client = create_client()

    document_context = format_documents(
        documents
    )

    prompt = f"""
{SYSTEM_INSTRUCTION}

===============================================================================
CUSTOMER-SUPPORT CONVERSATION
===============================================================================

{conversation}

===============================================================================
RETRIEVED TECHNICAL DOCUMENTATION
===============================================================================

{document_context}

===============================================================================
TASK
===============================================================================

Based strictly on the customer conversation and retrieved documentation,
generate a troubleshooting suggestion for the support agent.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty suggestion."
        )

    return response.text.strip()


# =============================================================================
# TEST DATA
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("RAG-GROUNDED SUPPORT SUGGESTION")
    print("=" * 80)
    print()

    conversation = """
Customer: The spindle keeps giving me overload warnings during machining.

Agent: When does the overload warning normally appear?

Customer: Mostly when I start the machining operation. Sometimes it happens
again when the spindle accelerates.

Agent: Does it happen with every machining program?

Customer: No, it seems to happen with this particular operation.
"""

    documents = [
        {
            "metadata": {
                "source": (
                    "english---hs-series-service-manual---2001.pdf"
                ),
                "start_page": 14,
                "end_page": 15,
                "section": (
                    "Does the spindle turn freely by hand?"
                ),
            },
            "text": """
Stalling / low torque generally relates to incorrect tooling or
machining practices.

A spindle that is tending to seize will yield a poor finish,
and run very hot and very loud.

Low line voltage may prevent the spindle from accelerating properly.

If the spindle takes a long time to accelerate, slows down or stays
at a speed below the commanded speed with the load meter at full load,
the spindle drive and motor are overloaded.

High load, low voltage, or too fast acceleration/deceleration can
cause this problem.
""",
        },
    ]

    suggestion = generate_suggestion(
        conversation=conversation,
        documents=documents,
    )

    print(suggestion)
    print()

    print("=" * 80)


if __name__ == "__main__":
    main()