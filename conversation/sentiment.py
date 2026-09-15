import os
import json

from dotenv import load_dotenv
from google import genai


load_dotenv()


MODEL_NAME = "gemini-3.6-flash"


client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def analyze_sentiment(conversation: str) -> dict:
    """
    Analyze the sentiment of the customer-support conversation.

    Returns:
        {
            "sentiment": "...",
            "confidence": "...",
            "reason": "..."
        }
    """

    prompt = f"""
You are analyzing a customer-support conversation for a technical support system.

Analyze the customer's overall sentiment based on the conversation.

Choose exactly one sentiment:

- Positive
- Neutral
- Calm
- Frustrated
- Agitated
- Angry
- Confused

Focus primarily on the customer's statements, not the support agent's tone.

Return ONLY valid JSON in this format:

{{
    "sentiment": "Frustrated",
    "confidence": "High",
    "reason": "Brief explanation based on the customer's statements."
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

    # Handle accidental markdown fences.
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return {
            "sentiment": "Unknown",
            "confidence": "Low",
            "reason": "Gemini returned an invalid JSON response.",
            "raw_response": text,
        }