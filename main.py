
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
import mimetypes
import os
import queue
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pyaudiowpatch as pyaudio
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

        self.client = None
        self.collector = None
        self.worker = None

        self.turns: list[dict[str, Any]] = []
        self.turn_counter = 0

        self.call_started_at = None

        self.subscribers: list[queue.Queue] = []


STATE = RuntimeState()


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

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
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

    return documents


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
                "Keeping latest conversation pending."
            )
            STATE.analysis_timer = threading.Timer(
                1.0,
                run_scheduled_ai_analysis,
            )
            STATE.analysis_timer.daemon = True
            STATE.analysis_timer.start()
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

        sentiment = analysis.get("sentiment", {})
        category = analysis.get("category", {})

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

    except Exception as error:
        print("Analysis error:", error)
        publish({
            "type": "error",
            "message": f"Analysis failed: {error}",
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

        publish({
            "type": "suggestion",
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

class Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format_string,
        *args,
    ):
        # Keep the terminal clean.
        return

    def _headers(
        self,
        content_type="application/json",
    ):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type,
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.send_header(
            "Cache-Control",
            "no-cache",
        )

        self.end_headers()

    def do_OPTIONS(self):

        self._headers()

    def do_GET(self):

        if self.path == "/health":

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

        if self.path == "/demo-audio":

            try:
                audio_path = get_demo_audio_path()
                data = audio_path.read_bytes()

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
            except Exception as error:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": str(error)
                }).encode("utf-8"))
            return

        if self.path == "/events":

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

            self.send_header(
                "Access-Control-Allow-Origin",
                "*",
            )

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

        if self.path not in (
            "/start",
            "/stop",
        ):

            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/start":

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
                STATE.last_ai_analysis_at = 0.0
                STATE.ai_processing = False
                STATE.analysis_pending = False

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
        "  http://127.0.0.1:5500/audio/frontend/index.html"
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

    server = ThreadingHTTPServer(
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
