import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import pyaudiowpatch as pyaudio

from dotenv import load_dotenv

from assemblyai.streaming.v3 import (
    SpeakerRevisionEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingEvents,
    StreamingParameters,
)


# =============================================================================
# Configuration
# =============================================================================

# AssemblyAI expects 16 kHz mono 16-bit PCM for this POC.
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2       # 16-bit PCM
CHANNELS = 1

# Send audio to AssemblyAI every 100 ms.
CHUNK_DURATION_MS = 100

SPEECH_MODEL = "universal-3-5-pro"

TRANSCRIPT_DIR = (
    Path("biesse_conversation_dataset")
    / "transcripts"
)


# =============================================================================
# WAV Validation
# =============================================================================

def validate_wav(wav_path: Path):

    if not wav_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {wav_path}"
        )

    with wave.open(str(wav_path), "rb") as wav:

        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()

        duration = frame_count / sample_rate

        print("Audio information:")
        print(f"  File          : {wav_path.name}")
        print(f"  Channels      : {channels}")
        print(f"  Sample rate   : {sample_rate} Hz")
        print(f"  Sample width  : {sample_width * 8}-bit")
        print(f"  Duration      : {duration:.2f} seconds")
        print()

        if channels != CHANNELS:
            raise ValueError(
                f"Expected mono audio, but file has "
                f"{channels} channels."
            )

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz audio, "
                f"but file is {sample_rate} Hz."
            )

        if sample_width != SAMPLE_WIDTH_BYTES:
            raise ValueError(
                "Expected 16-bit PCM audio."
            )

    return duration


# =============================================================================
# WAV Audio Chunks
# =============================================================================

def audio_chunks(wav_path: Path):
    """
    Read the WAV file in 100 ms chunks.

    AssemblyAI receives raw PCM audio.
    """

    frames_per_chunk = int(
        SAMPLE_RATE
        * CHUNK_DURATION_MS
        / 1000
    )

    with wave.open(str(wav_path), "rb") as wav:

        while True:

            data = wav.readframes(
                frames_per_chunk
            )

            if not data:
                break

            yield data


# =============================================================================
# WASAPI Loopback Audio
# =============================================================================

def loopback_audio_chunks():
    """
    Capture the computer's playback audio using WASAPI loopback.

    VLC / Windows Media Player
            ↓
    Windows playback device
            ↓
    WASAPI loopback
            ↓
    Python

    The loopback device may use a native format such as
    48 kHz stereo.

    AssemblyAI expects:
        16 kHz
        mono
        16-bit PCM

    Therefore this function converts the captured audio
    to the AssemblyAI format before yielding chunks.

    NOTE:
    This first implementation uses a simple linear
    resampling approach so that we do not introduce
    another audio-processing dependency.
    """

    with pyaudio.PyAudio() as p:

        loopback = p.get_default_wasapi_loopback()

        input_rate = int(
            loopback["defaultSampleRate"]
        )

        input_channels = int(
            loopback["maxInputChannels"]
        )

        if input_channels < 1:
            raise RuntimeError(
                "WASAPI loopback device has no input channels."
            )

        print()
        print("=" * 80)
        print("WASAPI LOOPBACK AUDIO")
        print("=" * 80)
        print(
            f"Device        : {loopback['name']}"
        )
        print(
            f"Device index  : {loopback['index']}"
        )
        print(
            f"Input rate    : {input_rate} Hz"
        )
        print(
            f"Input channels: {input_channels}"
        )
        print(
            f"Output format : {SAMPLE_RATE} Hz, "
            f"{CHANNELS} channel, 16-bit PCM"
        )
        print()

        # We capture roughly 100 ms at the native
        # playback-device sample rate.
        input_frames_per_chunk = int(
            input_rate
            * CHUNK_DURATION_MS
            / 1000
        )

        stream = p.open(
            format=pyaudio.paInt16,
            channels=input_channels,
            rate=input_rate,
            input=True,
            input_device_index=loopback["index"],
            frames_per_buffer=input_frames_per_chunk,
        )

        print(
            "WASAPI loopback capture started."
        )
        print(
            "Play your customer-support audio now."
        )
        print(
            "Press Ctrl+C to stop.\n"
        )

        try:

            while True:

                raw_data = stream.read(
                    input_frames_per_chunk,
                    exception_on_overflow=False,
                )

                # -------------------------------------------------------------
                # Convert raw PCM bytes → signed 16-bit samples
                # -------------------------------------------------------------

                sample_count = (
                    len(raw_data)
                    // SAMPLE_WIDTH_BYTES
                )

                if sample_count == 0:
                    continue

                import struct

                samples = struct.unpack(
                    "<"
                    + ("h" * sample_count),
                    raw_data,
                )

                # -------------------------------------------------------------
                # Stereo / multi-channel → mono
                # -------------------------------------------------------------

                if input_channels == 1:

                    mono_samples = list(samples)

                else:

                    mono_samples = []

                    frame_count = (
                        len(samples)
                        // input_channels
                    )

                    for frame_index in range(
                        frame_count
                    ):

                        frame_start = (
                            frame_index
                            * input_channels
                        )

                        frame = samples[
                            frame_start:
                            frame_start + input_channels
                        ]

                        # Average all channels.
                        value = sum(frame) / len(frame)

                        # Prevent integer overflow.
                        value = max(
                            -32768,
                            min(
                                32767,
                                int(value)
                            )
                        )

                        mono_samples.append(value)

                # -------------------------------------------------------------
                # Resample mono audio to 16 kHz
                # -------------------------------------------------------------

                if input_rate == SAMPLE_RATE:

                    output_samples = mono_samples

                else:

                    output_length = int(
                        len(mono_samples)
                        * SAMPLE_RATE
                        / input_rate
                    )

                    if output_length <= 0:
                        continue

                    output_samples = []

                    max_index = (
                        len(mono_samples) - 1
                    )

                    for output_index in range(
                        output_length
                    ):

                        source_position = (
                            output_index
                            * input_rate
                            / SAMPLE_RATE
                        )

                        left_index = int(
                            source_position
                        )

                        if left_index >= max_index:
                            output_samples.append(
                                mono_samples[-1]
                            )
                            continue

                        right_index = (
                            left_index + 1
                        )

                        fraction = (
                            source_position
                            - left_index
                        )

                        value = (
                            mono_samples[left_index]
                            * (1.0 - fraction)
                            +
                            mono_samples[right_index]
                            * fraction
                        )

                        value = max(
                            -32768,
                            min(
                                32767,
                                int(value)
                            )
                        )

                        output_samples.append(
                            value
                        )

                # -------------------------------------------------------------
                # Convert samples back to raw PCM bytes
                # -------------------------------------------------------------

                output_data = struct.pack(
                    "<"
                    + ("h" * len(output_samples)),
                    *output_samples,
                )

                if output_data:
                    yield output_data

        finally:

            stream.stop_stream()
            stream.close()

            print()
            print(
                "WASAPI loopback capture stopped."
            )


# =============================================================================
# Transcript Collector
# =============================================================================

class TranscriptCollector:

    def __init__(
        self,
        conversation_id: str,
        audio_path: Path,
    ):

        self.conversation_id = conversation_id
        self.audio_path = audio_path

        # Finalized AssemblyAI turns.
        self.turns = []

        # Our own sequential turn number.
        self.turn_counter = 0

        # AssemblyAI session ID.
        self.session_id = None

        # Whether SpeakerRevision was received.
        self.speaker_revision_received = False

    # -------------------------------------------------------------------------
    # Session started
    # -------------------------------------------------------------------------

    def on_begin(self, client, event):

        self.session_id = event.id

        print()
        print("=" * 80)
        print("ASSEMBLYAI STREAMING SESSION STARTED")
        print("=" * 80)

        print(
            f"Session ID: {self.session_id}"
        )

        print(
            f"Model: {SPEECH_MODEL}"
        )

        print(
            "Speaker diarization: enabled"
        )

        print(
            "Maximum speakers: 2"
        )

        print()

    # -------------------------------------------------------------------------
    # Transcript turn
    # -------------------------------------------------------------------------

    def on_turn(self, client, event):

        transcript = getattr(
            event,
            "transcript",
            None
        )

        if not transcript:
            return

        # AssemblyAI speaker label.
        speaker = (
            getattr(
                event,
                "speaker_label",
                None
            )
            or getattr(
                event,
                "speaker",
                None
            )
        )

        # -------------------------------------------------------------
        # Partial transcript
        # -------------------------------------------------------------

        if not event.end_of_turn:

            speaker_text = (
                f"[{speaker}] "
                if speaker
                else ""
            )

            print(
                f"\r[PARTIAL] "
                f"{speaker_text}"
                f"{transcript}",
                end="",
                flush=True,
            )

            return

        # -------------------------------------------------------------
        # Final transcript turn
        # -------------------------------------------------------------

        print()

        speaker_text = (
            f"[{speaker}] "
            if speaker
            else ""
        )

        print(
            f"[FINAL] "
            f"{speaker_text}"
            f"{transcript}"
        )

        # -------------------------------------------------------------
        # Store finalized turn
        # -------------------------------------------------------------

        self.turn_counter += 1

        # AssemblyAI event timing fields.
        start_ms = getattr(
            event,
            "start_ms",
            None
        )

        end_ms = getattr(
            event,
            "end_ms",
            None
        )

        turn = {
            "turn_id": self.turn_counter,
            "speaker": speaker,
            "text": transcript,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

        # Save word-level information when available.
        words = getattr(
            event,
            "words",
            None
        )

        if words:

            turn["words"] = [

                {
                    "text": getattr(
                        word,
                        "text",
                        ""
                    ),

                    "start_ms": getattr(
                        word,
                        "start",
                        None
                    ),

                    "end_ms": getattr(
                        word,
                        "end",
                        None
                    ),

                    "speaker": getattr(
                        word,
                        "speaker",
                        None
                    ),
                }

                for word in words
            ]

        self.turns.append(turn)

    # -------------------------------------------------------------------------
    # Speaker Revision
    # -------------------------------------------------------------------------

    def on_speaker_revision(
        self,
        client,
        event: SpeakerRevisionEvent,
    ):
        """
        Handle AssemblyAI's final speaker-label corrections.

        AssemblyAI's turn_order is ZERO-BASED:

            turn_order 0 -> self.turns[0]
            turn_order 1 -> self.turns[1]
            turn_order 2 -> self.turns[2]

        Our saved turn_id is ONE-BASED, but the list index is
        zero-based. Therefore we use turn_order directly.
        """

        self.speaker_revision_received = True

        print()
        print("=" * 80)
        print("SPEAKER REVISION RECEIVED")
        print("=" * 80)

        revisions = getattr(
            event,
            "revisions",
            []
        )

        print(
            f"Number of revisions: "
            f"{len(revisions)}"
        )

        print()
        print("Revision details:")

        for revision in revisions:

            turn_order = getattr(
                revision,
                "turn_order",
                None
            )

            revised_speaker = getattr(
                revision,
                "speaker_label",
                None
            )

            print()
            print(
                f"AssemblyAI turn_order: "
                f"{turn_order}"
            )

            print(
                f"  Revised speaker: "
                f"{revised_speaker}"
            )

            if (
                turn_order is None
                or turn_order < 0
                or turn_order >= len(self.turns)
            ):

                print(
                    "  Warning: turn could not "
                    "be matched to saved turns."
                )

                continue

            # IMPORTANT:
            # AssemblyAI turn_order is zero-based.
            index = turn_order

            old_speaker = self.turns[index][
                "speaker"
            ]

            # Apply corrected speaker label.
            self.turns[index][
                "speaker"
            ] = revised_speaker

            print(
                f"  Saved turn ID: "
                f"{self.turns[index]['turn_id']}"
            )

            print(
                f"  Old speaker: "
                f"{old_speaker}"
            )

            print(
                f"  New speaker: "
                f"{revised_speaker}"
            )

            print(
                f"  Text: "
                f"{self.turns[index]['text']}"
            )

            # Save revised word-level speaker information
            # if available.
            words = getattr(
                revision,
                "words",
                None
            )

            if words:

                self.turns[index][
                    "words"
                ] = [

                    {
                        "text": getattr(
                            word,
                            "text",
                            ""
                        ),

                        "start_ms": getattr(
                            word,
                            "start",
                            None
                        ),

                        "end_ms": getattr(
                            word,
                            "end",
                            None
                        ),

                        "speaker": getattr(
                            word,
                            "speaker",
                            None
                        ),
                    }

                    for word in words
                ]

        print()
        print(
            "Speaker revisions applied "
            "to saved turns."
        )

        print("=" * 80)
        print()

    # -------------------------------------------------------------------------
    # Error
    # -------------------------------------------------------------------------

    def on_error(self, client, error):

        print()
        print()
        print("=" * 80)
        print("ASSEMBLYAI ERROR")
        print("=" * 80)

        print(error)

    # -------------------------------------------------------------------------
    # Session terminated
    # -------------------------------------------------------------------------

    def on_terminated(self, client, event):

        print()
        print()
        print("=" * 80)
        print("ASSEMBLYAI STREAMING SESSION TERMINATED")
        print("=" * 80)

        print()

    # -------------------------------------------------------------------------
    # Save transcript
    # -------------------------------------------------------------------------

    def save(self, duration_seconds: float):

        TRANSCRIPT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            TRANSCRIPT_DIR
            / f"{self.conversation_id}_assemblyai.json"
        )

        output = {

            "conversation_id":
                self.conversation_id,

            "source_audio":
                str(self.audio_path),

            "duration_seconds":
                round(
                    duration_seconds,
                    2
                ),

            "transcription": {

                "provider":
                    "AssemblyAI",

                "streaming":
                    True,

                "speech_model":
                    SPEECH_MODEL,

                "sample_rate":
                    SAMPLE_RATE,

                "channels":
                    CHANNELS,

                "sample_width_bits":
                    SAMPLE_WIDTH_BYTES * 8,

                "chunk_duration_ms":
                    CHUNK_DURATION_MS,

                "speaker_labels":
                    True,

                "max_speakers":
                    2,

                "speaker_revision":
                    True,
            },

            "assemblyai_session": {

                "session_id":
                    self.session_id,
            },

            "diarization": {

                "speaker_revision_received":
                    self.speaker_revision_received,

                # Keep AssemblyAI's generic A/B labels.
                #
                # We do NOT assume:
                # A = customer
                # B = agent
                #
                # That mapping belongs to the
                # evaluation / processing layer.

                "label_type":
                    "assemblyai_generic_speaker_labels",
            },

            "turn_count":
                len(self.turns),

            "turns":
                self.turns,
        }

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            "Transcript saved to:"
        )

        print(
            f"  {output_path}"
        )

        print()

        return output_path


# =============================================================================
# AssemblyAI Client Creation
# =============================================================================

def create_assemblyai_client(collector):

    load_dotenv()

    api_key = os.getenv(
        "ASSEMBLYAI_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set "
            "in the environment."
        )

    client = StreamingClient(
        StreamingClientOptions(
            api_key=api_key,
            api_host="streaming.assemblyai.com",
        )
    )

    # -------------------------------------------------------------------------
    # Register callbacks
    # -------------------------------------------------------------------------

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

    return client


# =============================================================================
# AssemblyAI Streaming Connection
# =============================================================================

def connect_assemblyai(client):

    print(
        "Connecting to AssemblyAI..."
    )

    print()

    client.connect(
        StreamingParameters(

            speech_model=
                SPEECH_MODEL,

            sample_rate=
                SAMPLE_RATE,

            # Enable speaker diarization.
            speaker_labels=True,

            # Exactly two speakers:
            # customer + support agent.
            max_speakers=2,

            # Balanced latency/accuracy.
            mode="balanced",

            # Receive partial transcript updates.
            continuous_partials=True,
        )
    )


# =============================================================================
# WAV Streaming
# =============================================================================

def stream_wav(wav_path: Path):

    # -------------------------------------------------------------------------
    # Validate audio
    # -------------------------------------------------------------------------

    duration = validate_wav(
        wav_path
    )

    conversation_id = (
        wav_path.stem.upper()
    )

    collector = TranscriptCollector(
        conversation_id=conversation_id,
        audio_path=wav_path,
    )

    client = create_assemblyai_client(
        collector
    )

    try:

        connect_assemblyai(client)

        print(
            "Streaming WAV audio..."
        )

        print()

        chunk_duration_seconds = (
            CHUNK_DURATION_MS / 1000.0
        )

        for chunk in audio_chunks(
            wav_path
        ):

            client.stream(
                chunk
            )

            # Pace the prerecorded WAV so that
            # AssemblyAI receives it approximately
            # as a live stream.
            time.sleep(
                chunk_duration_seconds
            )

    except KeyboardInterrupt:

        print()
        print(
            "Streaming interrupted by user."
        )

    finally:

        print()
        print(
            "Closing AssemblyAI session..."
        )

        try:

            client.disconnect(
                terminate=True
            )

        except Exception as e:

            print(
                f"Warning while disconnecting: "
                f"{e}"
            )

    output_path = collector.save(
        duration_seconds=duration
    )

    return output_path


# =============================================================================
# WASAPI Loopback Streaming
# =============================================================================

def stream_loopback():

    """
    Stream the computer's live playback audio
    to AssemblyAI.

    Example:

        VLC
         ↓
        Windows playback
         ↓
        WASAPI loopback
         ↓
        Python
         ↓
        AssemblyAI
    """

    # There is no WAV file in this mode.
    # We use a descriptive source name in the
    # saved transcript.
    audio_path = Path(
        "WASAPI_LOOPBACK"
    )

    conversation_id = (
        "LOOPBACK"
    )

    collector = TranscriptCollector(
        conversation_id=conversation_id,
        audio_path=audio_path,
    )

    client = create_assemblyai_client(
        collector
    )

    start_time = time.time()

    try:

        connect_assemblyai(client)

        print(
            "Streaming WASAPI loopback audio "
            "to AssemblyAI..."
        )

        print()
        print(
            "================================================"
        )
        print(
            "PLAY YOUR CUSTOMER-SUPPORT AUDIO NOW"
        )
        print(
            "================================================"
        )
        print()
        print(
            "For example:"
        )
        print(
            "  VLC → C01.wav → Play"
        )
        print()
        print(
            "The application will listen to the "
            "computer playback audio."
        )
        print()
        print(
            "Press Ctrl+C after the conversation finishes."
        )
        print()

        # ---------------------------------------------------------------------
        # Continuous live audio
        # ---------------------------------------------------------------------

        for chunk in loopback_audio_chunks():

            client.stream(
                chunk
            )

    except KeyboardInterrupt:

        print()
        print(
            "Loopback streaming interrupted by user."
        )

    finally:

        print()
        print(
            "Closing AssemblyAI session..."
        )

        try:

            client.disconnect(
                terminate=True
            )

        except Exception as e:

            print(
                f"Warning while disconnecting: "
                f"{e}"
            )

    duration = time.time() - start_time

    output_path = collector.save(
        duration_seconds=duration
    )

    return output_path


# =============================================================================
# CLI
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Stream audio to AssemblyAI.\n\n"
            "WAV mode:\n"
            "  python audio\\\\assemblyai_stream.py C01.wav\n\n"
            "WASAPI loopback mode:\n"
            "  python audio\\\\assemblyai_stream.py --loopback"
        )
    )

    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the WAV audio file.",
    )

    parser.add_argument(
        "--loopback",
        action="store_true",
        help=(
            "Capture computer playback audio "
            "using WASAPI loopback."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Validate CLI mode
    # -------------------------------------------------------------------------

    if args.loopback and args.audio:

        parser.error(
            "Do not provide an audio file together "
            "with --loopback."
        )

    if not args.loopback and not args.audio:

        parser.error(
            "Provide a WAV file or use --loopback."
        )

    # -------------------------------------------------------------------------
    # Run selected mode
    # -------------------------------------------------------------------------

    try:

        if args.loopback:

            stream_loopback()

        else:

            wav_path = Path(
                args.audio
            )

            stream_wav(
                wav_path
            )

    except Exception as e:

        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)

        print(
            str(e)
        )

        print()

        sys.exit(1)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()