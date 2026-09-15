import os
import json

from dotenv import load_dotenv
from google import genai


load_dotenv()


MODEL_NAME = "gemini-3.6-flash"


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


CATEGORIES = [
    "Machine Operation",
    "Maintenance & Parts",
    "Technical Troubleshooting",
]


def categorize_query(conversation: str) -> dict:
    """
    Categorize the customer's technical support query.

    Returns:
        {
            "category": "...",
            "confidence": "...",
            "reason": "..."
        }
    """

    categories_text = "\n".join(
        f"- {category}"
        for category in CATEGORIES
    )

    prompt = f"""
You are categorizing a customer-support conversation
for a CNC machine technical support system.

Choose exactly ONE category from the following:

{categories_text}

Category definitions:

Machine Operation:
Questions about operating the machine, starting programs,
automatic mode, selecting programs, or normal machine operation.

Maintenance & Parts:
Questions about routine maintenance, replacement components,
filters, coolant maintenance, spare parts, or maintenance procedures.

Technical Troubleshooting:
Problems involving machine errors, warnings, unexpected behavior,
communication problems, spindle problems, or failures during operation.

Return ONLY valid JSON:

{{
    "category": "Technical Troubleshooting",
    "confidence": "High",
    "reason": "Brief explanation."
}}

Do not invent information.

Conversation:
{conversation}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        result = json.loads(text)

    except json.JSONDecodeError:
        return {
            "category": "Unknown",
            "confidence": "Low",
            "reason": "Gemini returned an invalid JSON response.",
            "raw_response": text,
        }

    # Safety check: Gemini must return one of our allowed categories.
    if result.get("category") not in CATEGORIES:
        result["category"] = "Unknown"
        result["confidence"] = "Low"

    return result