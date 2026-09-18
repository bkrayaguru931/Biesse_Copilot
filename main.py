
"""
Biesse AI Support Copilot - local integration server.

Architecture:
    VLC / Windows Media Player
        -> Windows playback device
        -> WASAPI loopback
        -> Python
        -> AssemblyAI Streaming
        -> live events
        -> browser dashboard

The browser UI is served separately by VS Code Live Server.
This Python process provides the local API + SSE event stream.

Run:
    python main.py

Then open:
    http://127.0.0.1:5500/audio/frontend/index.html

No call audio is written to disk by this server.
"""

from __future__ import annotations

import json
import hashlib
import mimetypes
import os
import queue
import re
import secrets
import sqlite3
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyaudiowpatch as pyaudio
import bcrypt
from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    SpeakerRevisionEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)

from google import genai
from google.genai import types

import chromadb


# =============================================================================
# Configuration
# =============================================================================

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

API_HOST = "127.0.0.1"
API_PORT = 8765
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://127.0.0.1:5500")
AUTH_DB_PATH = PROJECT_ROOT / "data" / "biesse_auth.db"
SESSION_TTL_SECONDS = 60 * 60 * 12

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CHUNK_DURATION_MS = 100

SPEECH_MODEL = "universal-3-5-pro"
GEMINI_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSIONS = 1536

CHROMA_PATH = (
    PROJECT_ROOT
    / "knowledge_base"
    / "chroma_db"
)

CHROMA_COLLECTION = "biesse_knowledge_base"

# For the current two-party demo.
# Change to B if a different live source maps B -> customer.
CUSTOMER_SPEAKER = os.getenv(
    "CUSTOMER_SPEAKER",
    "A",
).upper()

AGENT_SPEAKER = (
    "B" if CUSTOMER_SPEAKER == "A" else "A"
)


# =============================================================================
# Runtime state
# =============================================================================

class RuntimeState:

    def __init__(self):
        self.lock = threading.Lock()

        self.running = False
        self.stop_requested = False

        # Gemini quota protection:
        # A full AI analysis uses 2 generate_content requests
        # (analysis + suggestion). Keep at least 30 seconds between
        # analysis cycles and never run two cycles at the same time.
        self.last_ai_analysis_at = 0.0
        self.ai_processing = False

        # Conversation-aware AI scheduling.
        # We wait for a short pause before analyzing so Gemini sees
        # a meaningful portion of the conversation instead of one sentence.
        self.analysis_timer = None
        self.analysis_pending = False
        self.analysis_debounce_seconds = 4.0
        self.analysis_retry_count = 0
        self.max_analysis_retries = 3

        self.client = None
        self.collector = None
        self.worker = None

        self.turns: list[dict[str, Any]] = []
        self.turn_counter = 0

        self.call_started_at = None

        self.subscribers: list[queue.Queue] = []
        self.suggestion_contexts: dict[str, dict[str, Any]] = {}


STATE = RuntimeState()


# =============================================================================
# Local authentication
# =============================================================================

AUTH_LOCK = threading.Lock()
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def initialize_auth_db() -> None:
    """Create the small local account store used by this demonstration app."""
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS sessions_expiry_idx ON sessions(expires_at);
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                suggestion_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                document_key TEXT NOT NULL,
                source TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                conversation TEXT NOT NULL,
                suggestion_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(suggestion_id, user_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS feedback_document_idx
                ON recommendation_feedback(document_key);
            """
        )


def create_user(name: str, email: str, password: str) -> dict[str, Any]:
    name = name.strip()
    email = email.strip().lower()
    if not 2 <= len(name) <= 80:
        raise ValueError("Name must be between 2 and 80 characters.")
    if not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 10:
        raise ValueError("Password must contain at least 10 characters.")
    if len(password) > 128 or len(password.encode("utf-8")) > 72:
        raise ValueError("Password must be 72 bytes or fewer.")

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    try:
        with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
            cursor = connection.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, password_hash, int(time.time())),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError as error:
        raise ValueError("An account with that email already exists.") from error
    return {"id": user_id, "name": name, "email": email}


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    email = email.strip().lower()
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        row = connection.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
    if row is None or not bcrypt.checkpw(password.encode("utf-8"), row[3]):
        return None
    return {"id": row[0], "name": row[1], "email": row[2]}


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )
    return token


def get_session_user(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT users.id, users.name, users.email
            FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ? AND sessions.expires_at > ?
            """,
            (token_hash, int(time.time())),
        ).fetchone()
    return None if row is None else {"id": row[0], "name": row[1], "email": row[2]}


def delete_session(token: str | None) -> None:
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))


def document_key(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {})
    identity = "|".join(
        str(metadata.get(field, ""))
        for field in ("source", "start_page", "end_page", "section")
    )
    identity += "|" + document.get("text", "")[:300]
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def feedback_scores(document_keys: list[str]) -> dict[str, float]:
    """Return conservative feedback signals only after three independent votes."""
    if not document_keys:
        return {}
    placeholders = ", ".join("?" for _ in document_keys)
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        rows = connection.execute(
            f"""
            SELECT document_key, COUNT(*), SUM(rating)
            FROM recommendation_feedback
            WHERE document_key IN ({placeholders})
            GROUP BY document_key
            HAVING COUNT(*) >= 3
            """,
            document_keys,
        ).fetchall()
    return {key: max(-1.0, min(1.0, total / count)) for key, count, total in rows}


def store_feedback(user_id: int, suggestion_id: str, rating: int) -> bool:
    with STATE.lock:
        context = STATE.suggestion_contexts.get(suggestion_id)
    if context is None:
        raise ValueError("That recommendation is no longer available. Generate a new one and try again.")
    with AUTH_LOCK, sqlite3.connect(AUTH_DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO recommendation_feedback
            (suggestion_id, user_id, document_key, source, rating, conversation, suggestion_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (suggestion_id, user_id, context["document_key"], context["source"], rating,
             context["conversation"], json.dumps(context["suggestion"]), int(time.time())),
        )
    return cursor.rowcount > 0


# =============================================================================
# SSE event bus
# =============================================================================

def publish(event: dict[str, Any]):

    payload = json.dumps(
        event,
        ensure_ascii=False,
    )

    dead = []

    with STATE.lock:

        for subscriber in STATE.subscribers:

            try:
                subscriber.put_nowait(payload)

            except queue.Full:
                dead.append(subscriber)

        for subscriber in dead:

            if subscriber in STATE.subscribers:
                STATE.subscribers.remove(
                    subscriber
                )


def subscribe():

    q = queue.Queue(maxsize=100)

    with STATE.lock:
        STATE.subscribers.append(q)

    return q


def unsubscribe(q):

    with STATE.lock:

        if q in STATE.subscribers:
            STATE.subscribers.remove(q)


# =============================================================================
# Helpers
# =============================================================================

def elapsed_seconds() -> float:

    if STATE.call_started_at is None:
        return 0.0

    return time.time() - STATE.call_started_at


def speaker_role(label: str | None) -> str:

    if label == CUSTOMER_SPEAKER:
        return "customer"

    if label == AGENT_SPEAKER:
        return "agent"

    return "unknown"


def confidence_from_score(value: Any) -> str:

    try:
        score = float(value)

        if score >= 0.85:
            return "High"

        if score >= 0.65:
            return "Medium"

        return "Low"

    except Exception:
        return "Medium"


# =============================================================================
# Gemini / Chroma initialization
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set in .env"
    )

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = chroma_client.get_collection(
    name=CHROMA_COLLECTION
)

# Demo WAV played by the browser. Because the browser plays the WAV through
# the normal Windows playback device, WASAPI loopback captures the same audio
# and sends it to AssemblyAI.
DEMO_AUDIO_CANDIDATES = [
    PROJECT_ROOT / "biesse_conversation_dataset" / "audio" / "C01.wav",
    PROJECT_ROOT / "audio" / "biesse_conversation_dataset" / "audio" / "C01.wav",
    PROJECT_ROOT / "audio" / "C01.wav",
]


def get_demo_audio_path() -> Path:
    for path in DEMO_AUDIO_CANDIDATES:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Demo audio C01.wav was not found. Checked: "
        + ", ".join(str(p) for p in DEMO_AUDIO_CANDIDATES)
    )


# =============================================================================
# RAG
# =============================================================================

def embed_query(query: str) -> list[float]:

    response = gemini_client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "task: question answering | "
                            f"query: {query}"
                        )
                    )
                ],
            )
        ],

        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )

    return response.embeddings[0].values


def rag_search(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:

    embedding = embed_query(query)

    # Retrieve extra candidates first. Feedback is used only as a small,
    # evidence-based reranking signal; semantic similarity remains primary.
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k * 3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = []

    raw_docs = (
        result.get("documents", [[]])[0]
    )

    raw_metadata = (
        result.get("metadatas", [[]])[0]
    )

    raw_distances = (
        result.get("distances", [[]])[0]
    )

    for i, document in enumerate(raw_docs):

        metadata = (
            raw_metadata[i]
            if i < len(raw_metadata)
            else {}
        )

        distance = (
            raw_distances[i]
            if i < len(raw_distances)
            else None
        )

        documents.append({

            "text": document,

            "metadata": metadata,

            "distance": distance,
        })

    scores = feedback_scores([document_key(document) for document in documents])
    for document in documents:
        score = scores.get(document_key(document), 0.0)
        document["feedback_score"] = score
        distance = document.get("distance")
        # Chroma distances are lower-is-better. Cap feedback influence so a
        # historically popular result cannot override a clearly better match.
        document["reranked_distance"] = (distance if distance is not None else float("inf")) - (0.12 * score)

    documents.sort(key=lambda document: document["reranked_distance"])
    return documents[:top_k]


# =============================================================================
# Gemini analysis
# =============================================================================

ANALYSIS_INSTRUCTION = """
You are an AI copilot assisting a technical customer-support agent
during a live machine-support call.

Analyze the conversation so far.

Return JSON only:

{
  "sentiment": {
    "label": "Positive | Neutral | Negative | Agitated | Frustrated",
    "reason": "short explanation",
    "confidence": 0.0
  },
  "category": {
    "label": "Machine Operation | Maintenance & Parts | Technical Troubleshooting",
    "reason": "short explanation",
    "confidence": 0.0
  }
}

Rules:
- Analyze the CUSTOMER tone, not the agent's tone.
- Use the whole conversation, not only the latest sentence.
- Do not invent facts.
- Keep reasons concise.
"""


def analyze_conversation(
    conversation_text: str,
) -> dict[str, Any]:

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=(
            ANALYSIS_INSTRUCTION
            + "\n\nCONVERSATION:\n"
            + conversation_text
        ),

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(
                text[start:end + 1]
            )

        raise


# =============================================================================
# Gemini suggestion generation
# =============================================================================

SUGGESTION_INSTRUCTION = """
You are a technical support copilot for a machine-support agent.

Use the conversation and retrieved technical documentation to help
the agent respond to the customer.

Return JSON only:

{
  "summary": "2-3 sentence concise assessment",
  "actions": [
    "action 1",
    "action 2",
    "action 3"
  ],
  "questions": [
    "question 1",
    "question 2",
    "question 3"
  ]
}

Rules:
- Retrieved documents are the primary technical source.
- Do not claim generic manuals are Biesse-specific.
- Do not invent machine-specific alarm codes, part numbers,
  settings, or procedures.
- Do not state a root cause as certain unless the documentation
  supports it.
- Present possible causes as possibilities.
- Prefer diagnostic checks and questions before replacement.
- Do not recommend unsafe actions.
- If the documentation is insufficient, explicitly say that more
  information is required.
"""


def generate_suggestion(
    conversation_text: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:

    context_parts = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        metadata = document.get(
            "metadata",
            {},
        )

        context_parts.append(

            f"DOCUMENT {index}\n"
            f"Source: {metadata.get('source', 'Unknown')}\n"
            f"Pages: "
            f"{metadata.get('start_page', '?')}-"
            f"{metadata.get('end_page', '?')}\n"
            f"Section: "
            f"{metadata.get('section', 'Unknown')}\n"
            f"Content:\n"
            f"{document.get('text', '')}"
        )

    retrieved_context = "\n\n".join(
        context_parts
    )

    response = gemini_client.models.generate_content(

        model=GEMINI_MODEL,

        contents=(
            SUGGESTION_INSTRUCTION
            + "\n\nCONVERSATION:\n"
            + conversation_text
            + "\n\nRETRIEVED DOCUMENTATION:\n"
            + retrieved_context
        ),

        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = response.text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(
                text[start:end + 1]
            )

        raise


# =============================================================================
# Conversation processing
# =============================================================================

def conversation_text() -> str:

    with STATE.lock:

        turns = list(STATE.turns)

    lines = []

    for turn in turns:

        role = turn.get(
            "role",
            "unknown",
        )

        name = (
            "Customer"
            if role == "customer"
            else "Agent"
        )

        lines.append(
            f"{name}: {turn.get('text', '')}"
        )

    return "\n".join(lines)


def should_process_customer_turn(
    text: str,
) -> bool:

    if len(text.strip()) < 12:
        return False

    technical_terms = [

        "machine",
        "spindle",
        "overload",
        "alarm",
        "warning",
        "program",
        "cycle",
        "automatic",
        "maintenance",
        "filter",
        "coolant",
        "tool",
        "error",
        "fault",
        "communication",
        "network",
        "voltage",
        "motor",
    ]

    lower = text.lower()

    return any(
        term in lower
        for term in technical_terms
    )


def request_ai_analysis(turn: dict[str, Any]):
    """Schedule analysis after the conversation has been quiet briefly."""

    # Only customer turns can trigger an AI analysis.
    if turn.get("role") != "customer":
        return

    if not should_process_customer_turn(turn.get("text", "")):
        return

    with STATE.lock:
        if not STATE.running:
            return

        # Remember that the conversation changed.
        STATE.analysis_pending = True

        # Every new customer turn resets the debounce timer.
        if STATE.analysis_timer is not None:
            STATE.analysis_timer.cancel()

        STATE.analysis_timer = threading.Timer(
            STATE.analysis_debounce_seconds,
            run_scheduled_ai_analysis,
        )
        STATE.analysis_timer.daemon = True
        STATE.analysis_timer.start()

    print(
        f"[AI] Analysis scheduled in "
        f"{STATE.analysis_debounce_seconds:.1f}s."
    )


def run_scheduled_ai_analysis():
    """Run the latest conversation through the AI pipeline when ready."""

    with STATE.lock:
        if not STATE.running or not STATE.analysis_pending:
            return

        # If another AI cycle is running, keep the pending flag.
        if STATE.ai_processing:
            print(
                "[AI] Analysis already running. "
                "Latest conversation will run when it completes."
            )
            # The active cycle's finally block sees analysis_pending and
            # schedules exactly one follow-up. Polling here created a stream
            # of redundant timers and noisy logs while generation was slow.
            return

        # Respect the 30-second Gemini generate_content cooldown.
        now = time.time()
        if (
            STATE.last_ai_analysis_at > 0
            and now - STATE.last_ai_analysis_at < 30
        ):
            remaining = 30 - (now - STATE.last_ai_analysis_at)
            print(
                f"[AI] Gemini cooldown active. "
                f"Retrying in {remaining:.1f}s."
            )
            STATE.analysis_timer = threading.Timer(
                max(remaining, 0.5),
                run_scheduled_ai_analysis,
            )
            STATE.analysis_timer.daemon = True
            STATE.analysis_timer.start()
            return

        STATE.analysis_pending = False
        STATE.ai_processing = True
        STATE.last_ai_analysis_at = now

    try:
        print("[AI] Starting analysis of latest conversation.")
        _run_ai_analysis()
    finally:
        with STATE.lock:
            STATE.ai_processing = False
            pending = STATE.analysis_pending
            running = STATE.running

        # If new customer turns arrived while Gemini was working,
        # analyze the newest conversation after the normal debounce.
        if pending and running:
            print(
                "[AI] Conversation changed during analysis. "
                "Scheduling another analysis."
            )
            with STATE.lock:
                if STATE.analysis_timer is not None:
                    STATE.analysis_timer.cancel()
                STATE.analysis_timer = threading.Timer(
                    STATE.analysis_debounce_seconds,
                    run_scheduled_ai_analysis,
                )
                STATE.analysis_timer.daemon = True
                STATE.analysis_timer.start()


def _run_ai_analysis():
    """Analyze the latest complete conversation using Gemini + RAG."""

    text = conversation_text()
    if not text.strip():
        return

    # -------------------------------------------------------------------------
    # Sentiment + category
    # -------------------------------------------------------------------------
    try:
        analysis = analyze_conversation(text)
        if not isinstance(analysis, dict):
            raise ValueError("Gemini analysis response was not a JSON object.")

        sentiment = analysis.get("sentiment", {})
        category = analysis.get("category", {})
        sentiment = sentiment if isinstance(sentiment, dict) else {}
        category = category if isinstance(category, dict) else {}

        sentiment_payload = {
            "label": sentiment.get("label", "Neutral"),
            "reason": sentiment.get("reason", ""),
            "confidence": confidence_from_score(
                sentiment.get("confidence", 0.7)
            ),
        }

        category_payload = {
            "label": category.get(
                "label",
                "Technical Troubleshooting",
            ),
            "reason": category.get("reason", ""),
            "confidence": confidence_from_score(
                category.get("confidence", 0.7)
            ),
        }

        publish({
            "type": "analysis",
            "sentiment": sentiment_payload,
            "category": category_payload,
        })

        with STATE.lock:
            STATE.analysis_retry_count = 0

    except Exception as error:
        is_temporary_provider_error = (
            getattr(error, "code", None) in (429, 500, 502, 503, 504)
            or any(code in str(error) for code in ("429", "500", "502", "503", "504"))
        )
        with STATE.lock:
            if (
                is_temporary_provider_error
                and STATE.analysis_retry_count < STATE.max_analysis_retries
                and STATE.running
            ):
                STATE.analysis_retry_count += 1
                STATE.analysis_pending = True
                retry_number = STATE.analysis_retry_count
            else:
                retry_number = 0

        if retry_number:
            print(
                "[AI] Gemini is temporarily unavailable. "
                f"Retry {retry_number}/{STATE.max_analysis_retries} will run after the cooldown."
            )
        else:
            print("Analysis error:", error)
            publish({
                "type": "error",
                "message": "Conversation analysis is temporarily unavailable. Suggestions will continue.",
            })
        # Continue to RAG. Sentiment/category failure should not
        # prevent a technical suggestion from being generated.
        print("[AI] Continuing to RAG despite analysis failure.")

    # -------------------------------------------------------------------------
    # RAG + suggestion
    # -------------------------------------------------------------------------
    try:
        started = time.perf_counter()

        documents = rag_search(
            query=text,
            top_k=3,
        )

        if not documents:
            print("[RAG] No relevant documents found.")
            publish({
                "type": "error",
                "message":
                    "No relevant technical documentation found.",
            })
            return

        suggestion = generate_suggestion(
            conversation_text=text,
            documents=documents,
        )
        if not isinstance(suggestion, dict):
            raise ValueError("Gemini suggestion response was not a JSON object.")
        suggestion = {
            "summary": str(suggestion.get("summary", "")),
            "actions": [str(item) for item in suggestion.get("actions", [])][:5]
                if isinstance(suggestion.get("actions"), list) else [],
            "questions": [str(item) for item in suggestion.get("questions", [])][:5]
                if isinstance(suggestion.get("questions"), list) else [],
        }

        latency = time.perf_counter() - started

        top_document = documents[0]
        metadata = top_document.get("metadata", {})

        source = metadata.get(
            "source",
            "Technical reference",
        )

        start_page = metadata.get("start_page")
        end_page = metadata.get("end_page")

        if start_page and end_page:
            pages = f"{start_page}-{end_page}"
        elif start_page:
            pages = str(start_page)
        else:
            pages = ""

        suggestion_id = str(uuid.uuid4())
        with STATE.lock:
            STATE.suggestion_contexts[suggestion_id] = {
                "document_key": document_key(top_document),
                "source": source,
                "conversation": text,
                "suggestion": suggestion,
            }
            # Keep only the most recent recommendations in memory. Feedback is
            # persisted separately, so expired UI actions do not grow memory.
            while len(STATE.suggestion_contexts) > 100:
                STATE.suggestion_contexts.pop(next(iter(STATE.suggestion_contexts)))

        publish({
            "type": "suggestion",
            "suggestion_id": suggestion_id,
            "summary": suggestion.get("summary", ""),
            "actions": suggestion.get("actions", []),
            "questions": suggestion.get("questions", []),
            "source": source,
            "pages": pages,
            "latency_seconds": round(latency, 2),
        })

        print(
            f"[AI] Suggestion generated in {latency:.2f}s."
        )

    except Exception as error:
        print("RAG/suggestion error:", error)
        publish({
            "type": "error",
            "message": f"Suggestion generation failed: {error}",
        })

# =============================================================================
# AssemblyAI collector
# =============================================================================

class LiveCollector:

    def on_begin(
        self,
        client,
        event,
    ):

        print(
            "AssemblyAI session started:",
            event.id,
        )

        publish({

            "type": "status",

            "running": True,

            "message":
                "AssemblyAI streaming started.",
        })

    def on_turn(
        self,
        client,
        event,
    ):

        transcript = getattr(
            event,
            "transcript",
            None,
        )

        if not transcript:
            return

        speaker = (

            getattr(
                event,
                "speaker_label",
                None,
            )

            or getattr(
                event,
                "speaker",
                None,
            )
        )

        role = speaker_role(
            speaker
        )

        # -------------------------------------------------------------
        # Partial transcript
        # -------------------------------------------------------------

        if not event.end_of_turn:

            publish({

                "type": "partial",

                "speaker":
                    role,

                "text":
                    transcript,

                "elapsed_seconds":
                    round(
                        elapsed_seconds(),
                        2,
                    ),
            })

            return

        # -------------------------------------------------------------
        # Final transcript turn
        # -------------------------------------------------------------

        with STATE.lock:

            STATE.turn_counter += 1

            turn = {

                "turn_id":
                    STATE.turn_counter,

                "speaker":
                    speaker,

                "role":
                    role,

                "text":
                    transcript,

                "start_ms":
                    getattr(
                        event,
                        "start_ms",
                        None,
                    ),

                "end_ms":
                    getattr(
                        event,
                        "end_ms",
                        None,
                    ),
            }

            STATE.turns.append(
                turn
            )

        print(
            f"[FINAL] "
            f"[{speaker}/{role}] "
            f"{transcript}"
        )

        publish({

            "type": "turn",

            "turn_id":
                turn["turn_id"],

            "speaker":
                role,

            "assemblyai_speaker":
                speaker,

            "text":
                transcript,

            "elapsed_seconds":
                round(
                    elapsed_seconds(),
                    2,
                ),
        })

        # Do AI processing outside the AssemblyAI
        # callback so a Gemini/RAG request cannot
        # block incoming transcription events.
        # Schedule conversation-aware AI analysis.
        # New customer turns reset the debounce timer instead of
        # launching another Gemini request immediately.
        request_ai_analysis(turn)

    def on_speaker_revision(
        self,
        client,
        event: SpeakerRevisionEvent,
    ):

        revisions = getattr(
            event,
            "revisions",
            [],
        )

        print(
            "SpeakerRevision:",
            len(revisions),
            "revision(s)",
        )

        for revision in revisions:

            turn_order = getattr(
                revision,
                "turn_order",
                None,
            )

            revised_speaker = getattr(
                revision,
                "speaker_label",
                None,
            )

            if (
                turn_order is None
                or turn_order < 0
            ):
                continue

            with STATE.lock:

                if turn_order >= len(
                    STATE.turns
                ):
                    continue

                turn = STATE.turns[
                    turn_order
                ]

                turn["speaker"] = (
                    revised_speaker
                )

                turn["role"] = (
                    speaker_role(
                        revised_speaker
                    )
                )

            publish({

                "type":
                    "speaker_revision",

                "turn_id":
                    turn.get("turn_id"),

                "speaker":
                    speaker_role(
                        revised_speaker
                    ),

                "assemblyai_speaker":
                    revised_speaker,
            })

    def on_error(
        self,
        client,
        error,
    ):

        print(
            "AssemblyAI error:",
            error,
        )

        publish({

            "type": "error",

            "message":
                f"AssemblyAI error: {error}",
        })

    def on_terminated(
        self,
        client,
        event,
    ):

        print(
            "AssemblyAI session terminated."
        )


# =============================================================================
# WASAPI audio conversion
# =============================================================================

def loopback_audio_chunks():

    with pyaudio.PyAudio() as p:

        loopback = (
            p.get_default_wasapi_loopback()
        )

        input_rate = int(
            loopback[
                "defaultSampleRate"
            ]
        )

        input_channels = int(
            loopback[
                "maxInputChannels"
            ]
        )

        frames_per_chunk = int(
            input_rate
            * CHUNK_DURATION_MS
            / 1000
        )

        print()
        print(
            "WASAPI loopback:",
            loopback["name"],
        )

        print(
            "Native format:",
            input_rate,
            "Hz /",
            input_channels,
            "channel(s)",
        )

        stream = p.open(

            format=pyaudio.paInt16,

            channels=input_channels,

            rate=input_rate,

            input=True,

            input_device_index=
                loopback["index"],

            frames_per_buffer=
                frames_per_chunk,
        )

        try:

            while True:

                with STATE.lock:

                    if (
                        STATE.stop_requested
                        or not STATE.running
                    ):
                        break

                raw_data = stream.read(

                    frames_per_chunk,

                    exception_on_overflow=False,
                )

                sample_count = (
                    len(raw_data)
                    // SAMPLE_WIDTH_BYTES
                )

                if sample_count <= 0:
                    continue

                samples = struct.unpack(

                    "<"
                    + (
                        "h"
                        * sample_count
                    ),

                    raw_data,
                )

                # -------------------------------------------------------------
                # Downmix to mono
                # -------------------------------------------------------------

                if input_channels == 1:

                    mono = list(
                        samples
                    )

                else:

                    mono = []

                    frame_count = (
                        len(samples)
                        // input_channels
                    )

                    for frame_index in range(
                        frame_count
                    ):

                        start = (
                            frame_index
                            * input_channels
                        )

                        frame = samples[
                            start:
                            start + input_channels
                        ]

                        value = int(
                            sum(frame)
                            / len(frame)
                        )

                        value = max(
                            -32768,
                            min(
                                32767,
                                value,
                            ),
                        )

                        mono.append(
                            value
                        )

                # -------------------------------------------------------------
                # Resample to 16 kHz
                # -------------------------------------------------------------

                if input_rate == SAMPLE_RATE:

                    output = mono

                else:

                    output_length = int(

                        len(mono)
                        * SAMPLE_RATE
                        / input_rate
                    )

                    if output_length <= 0:
                        continue

                    output = []

                    max_index = (
                        len(mono) - 1
                    )

                    for index in range(
                        output_length
                    ):

                        source_position = (

                            index
                            * input_rate
                            / SAMPLE_RATE
                        )

                        left = int(
                            source_position
                        )

                        if left >= max_index:

                            output.append(
                                mono[-1]
                            )

                            continue

                        right = left + 1

                        fraction = (
                            source_position
                            - left
                        )

                        value = (

                            mono[left]
                            * (1.0 - fraction)

                            +

                            mono[right]
                            * fraction
                        )

                        output.append(
                            max(
                                -32768,
                                min(
                                    32767,
                                    int(value),
                                ),
                            )
                        )

                yield struct.pack(

                    "<"
                    + (
                        "h"
                        * len(output)
                    ),

                    *output,
                )

        finally:

            stream.stop_stream()
            stream.close()

            print(
                "WASAPI loopback stopped."
            )


# =============================================================================
# AssemblyAI streaming worker
# =============================================================================

def run_live_session():

    collector = LiveCollector()

    api_key = os.getenv(
        "ASSEMBLYAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set."
        )

    client = StreamingClient(

        StreamingClientOptions(

            api_key=api_key,

            api_host=
                "streaming.assemblyai.com",
        )
    )

    client.on(
        StreamingEvents.Begin,
        collector.on_begin,
    )

    client.on(
        StreamingEvents.Turn,
        collector.on_turn,
    )

    client.on(
        StreamingEvents.SpeakerRevision,
        collector.on_speaker_revision,
    )

    client.on(
        StreamingEvents.Error,
        collector.on_error,
    )

    client.on(
        StreamingEvents.Termination,
        collector.on_terminated,
    )

    with STATE.lock:

        STATE.client = client

    client.connect(

        StreamingParameters(

            speech_model=
                SPEECH_MODEL,

            sample_rate=
                SAMPLE_RATE,

            speaker_labels=True,

            max_speakers=2,

            mode="balanced",

            continuous_partials=True,
        )
    )

    try:

        for chunk in (
            loopback_audio_chunks()
        ):

            with STATE.lock:

                if (
                    STATE.stop_requested
                    or not STATE.running
                ):
                    break

            client.stream(
                chunk
            )

    finally:

        try:

            client.disconnect(
                terminate=True
            )

        except Exception as error:

            print(
                "Disconnect warning:",
                error,
            )

        with STATE.lock:

            STATE.client = None
            STATE.running = False
            STATE.stop_requested = False

        publish({

            "type": "stopped",
        })


# =============================================================================
# HTTP server
# =============================================================================

class CopilotHTTPServer(ThreadingHTTPServer):
    """Avoid noisy tracebacks when a browser closes an SSE connection."""

    def handle_error(self, request, client_address):
        import sys

        error_type, error, _ = sys.exc_info()
        if error_type is ConnectionAbortedError or isinstance(error, ConnectionAbortedError):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format_string,
        *args,
    ):
        # Keep the terminal clean.
        return

    def _headers(self, content_type="application/json", status=200):
        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type,
        )

        origin = self.headers.get("Origin")
        if origin == FRONTEND_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", origin)

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )

        self.send_header(
            "Cache-Control",
            "no-cache",
        )

        self.end_headers()

    def _json(self, payload: dict[str, Any], status=200):
        self._headers(status=status)
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _request_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request body.") from error
        if length <= 0 or length > 16_384:
            raise ValueError("Request body is missing or too large.")
        try:
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object.")
        return payload

    def _token(self) -> str | None:
        authorization = self.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            return authorization[7:].strip()
        return parse_qs(urlparse(self.path).query).get("token", [None])[0]

    def _current_user(self) -> dict[str, Any] | None:
        return get_session_user(self._token())

    def _require_user(self) -> dict[str, Any] | None:
        user = self._current_user()
        if user is None:
            self._json({"message": "Authentication is required."}, status=401)
        return user

    def do_OPTIONS(self):

        self._headers()

    def do_GET(self):

        request_path = urlparse(self.path).path

        if request_path == "/health":

            self._headers()

            payload = json.dumps({

                "status":
                    "ok",

                "running":
                    STATE.running,

                "customer_speaker":
                    CUSTOMER_SPEAKER,

                "agent_speaker":
                    AGENT_SPEAKER,
            }).encode("utf-8")

            self.wfile.write(
                payload
            )

            return

        if request_path == "/me":
            user = self._require_user()
            if user is not None:
                self._json({"user": user})
            return

        if request_path in ("/demo-audio", "/events") and self._require_user() is None:
            return

        if request_path == "/demo-audio":

            try:
                audio_path = get_demo_audio_path()
                data = audio_path.read_bytes()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                origin = self.headers.get("Origin")
                if origin == FRONTEND_ORIGIN:
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                origin = self.headers.get("Origin")
                if origin == FRONTEND_ORIGIN:
                    self.send_header("Access-Control-Allow-Origin", origin)
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": str(error)
                }).encode("utf-8"))
            return

        if request_path == "/events":

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/event-stream",
            )

            self.send_header(
                "Cache-Control",
                "no-cache",
            )

            self.send_header(
                "Connection",
                "keep-alive",
            )

            origin = self.headers.get("Origin")
            if origin == FRONTEND_ORIGIN:
                self.send_header("Access-Control-Allow-Origin", origin)

            self.end_headers()

            subscriber = subscribe()

            try:

                # Immediately tell the UI the backend is alive.
                self.wfile.write(
                    b"data: "
                    + json.dumps({
                        "type": "ready"
                    }).encode("utf-8")
                    + b"\n\n"
                )

                self.wfile.flush()

                while True:

                    try:

                        payload = (
                            subscriber.get(
                                timeout=15
                            )
                        )

                        message = (
                            "data: "
                            + payload
                            + "\n\n"
                        )

                        self.wfile.write(
                            message.encode(
                                "utf-8"
                            )
                        )

                        self.wfile.flush()

                    except queue.Empty:

                        # SSE heartbeat.
                        self.wfile.write(
                            b": heartbeat\n\n"
                        )

                        self.wfile.flush()

            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):

                pass

            finally:

                unsubscribe(
                    subscriber
                )

            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):

        request_path = urlparse(self.path).path

        if request_path in ("/auth/signup", "/auth/login"):
            try:
                payload = self._request_json()
                email = str(payload.get("email", ""))
                password = str(payload.get("password", ""))
                if request_path == "/auth/signup":
                    user = create_user(str(payload.get("name", "")), email, password)
                else:
                    user = authenticate_user(email, password)
                    if user is None:
                        self._json({"message": "Invalid email or password."}, status=401)
                        return
                self._json({"user": user, "token": create_session(user["id"])}, status=201 if request_path.endswith("signup") else 200)
            except ValueError as error:
                self._json({"message": str(error)}, status=400)
            return

        if request_path == "/auth/logout":
            if self._require_user() is not None:
                delete_session(self._token())
                self._json({"status": "logged_out"})
            return

        if request_path == "/feedback":
            user = self._require_user()
            if user is None:
                return
            try:
                payload = self._request_json()
                rating_name = payload.get("rating")
                rating = {"helpful": 1, "not_helpful": -1}.get(rating_name)
                suggestion_id = str(payload.get("suggestion_id", ""))
                if rating is None or not suggestion_id:
                    raise ValueError("A recommendation and valid rating are required.")
                saved = store_feedback(user["id"], suggestion_id, rating)
                self._json({"status": "saved" if saved else "already_recorded"})
            except ValueError as error:
                self._json({"message": str(error)}, status=400)
            return

        if request_path not in (
            "/start",
            "/stop",
        ):

            self.send_response(404)
            self.end_headers()
            return

        if self._require_user() is None:
            return

        if request_path == "/start":

            with STATE.lock:

                if STATE.running:

                    self._headers()

                    self.wfile.write(
                        json.dumps({
                            "status":
                                "already_running"
                        }).encode()
                    )

                    return

                STATE.running = True
                STATE.stop_requested = False

                STATE.turns = []
                STATE.turn_counter = 0
                STATE.suggestion_contexts = {}
                STATE.last_ai_analysis_at = 0.0
                STATE.ai_processing = False
                STATE.analysis_pending = False
                STATE.analysis_retry_count = 0

                if STATE.analysis_timer is not None:
                    STATE.analysis_timer.cancel()
                STATE.analysis_timer = None

                STATE.call_started_at = (
                    time.time()
                )

            publish({

                "type": "status",

                "running": True,

                "message":
                    "Starting live call.",
            })

            worker = threading.Thread(

                target=self._start_worker,

                daemon=True,
            )

            with STATE.lock:
                STATE.worker = worker

            worker.start()

            self._headers()

            self.wfile.write(
                json.dumps({
                    "status": "started"
                }).encode()
            )

            return

        # ---------------------------------------------------------------------
        # STOP
        # ---------------------------------------------------------------------

        with STATE.lock:

            STATE.stop_requested = True

            if STATE.analysis_timer is not None:
                STATE.analysis_timer.cancel()

            STATE.analysis_timer = None
            STATE.analysis_pending = False

            client = STATE.client

        if client is not None:

            try:
                client.disconnect(
                    terminate=True
                )

            except Exception:
                pass

        publish({

            "type": "status",

            "running": False,

            "message":
                "Stopping live call.",
        })

        self._headers()

        self.wfile.write(
            json.dumps({
                "status": "stopping"
            }).encode()
        )

    @staticmethod
    def _start_worker():

        try:

            run_live_session()

        except Exception as error:

            print(
                "Live session error:",
                error,
            )

            with STATE.lock:

                STATE.running = False
                STATE.stop_requested = False

            publish({

                "type": "error",

                "message":
                    str(error),
            })


# =============================================================================
# Main
# =============================================================================

def main():

    initialize_auth_db()

    print()
    print("=" * 80)
    print("BIESSE AI SUPPORT COPILOT")
    print("=" * 80)
    print()
    print(
        f"Backend: http://{API_HOST}:{API_PORT}"
    )
    print(
        "Frontend:"
    )
    print(
        "  http://127.0.0.1:5500/frontend/index.html"
    )
    print()
    print(
        f"Customer speaker label: {CUSTOMER_SPEAKER}"
    )
    print(
        "Gemini model: "
        f"{GEMINI_MODEL}"
    )
    print(
        "Gemini AI analysis cooldown: 30 seconds"
    )
    try:
        print(
            "Demo audio:",
            get_demo_audio_path()
        )
    except Exception as error:
        print(
            "Demo audio warning:",
            error
        )
    print(
        f"Agent speaker label   : {AGENT_SPEAKER}"
    )
    print()
    print(
        "Waiting for the browser to start a call..."
    )
    print()

    server = CopilotHTTPServer(
        (
            API_HOST,
            API_PORT,
        ),
        Handler,
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print(
            "Stopping backend..."
        )

    finally:

        with STATE.lock:

            STATE.stop_requested = True
            client = STATE.client

        if client is not None:

            try:

                client.disconnect(
                    terminate=True
                )

            except Exception:
                pass

        server.server_close()


if __name__ == "__main__":
    main()
