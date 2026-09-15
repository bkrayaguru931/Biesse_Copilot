import asyncio
import io
import json
import sys
import wave
from pathlib import Path

import edge_tts
import miniaudio


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CONVERSATIONS_DIR = BASE_DIR / "conversations"
AUDIO_DIR = BASE_DIR / "audio"

# Different voices for speaker separation
CUSTOMER_VOICE = "en-US-AriaNeural"
AGENT_VOICE = "en-US-GuyNeural"

# Pause between Customer and Agent
PAUSE_MS = 500

# Final WAV format
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM


# ============================================================
# VOICE SELECTION
# ============================================================

def get_voice(speaker: str) -> str:
    """
    Return the Edge TTS voice for a speaker.
    """

    speaker = speaker.lower().strip()

    if speaker == "customer":
        return CUSTOMER_VOICE

    if speaker == "agent":
        return AGENT_VOICE

    raise ValueError(
        f"Unknown speaker '{speaker}'. "
        f"Expected 'Customer' or 'Agent'."
    )


# ============================================================
# SPEECH PARAMETERS
# ============================================================

def get_speech_parameters(
    speaker: str,
    tone: str
):
    """
    Apply subtle speech-rate changes to the customer's voice
    based on the scenario tone.

    Agent remains consistent and professional.
    """

    if speaker.lower() != "customer":
        return "+0%", "+0Hz"

    tone = tone.lower().strip()

    if tone == "angry":
        return "+8%", "+0Hz"

    if tone == "agitated":
        return "+7%", "+0Hz"

    if tone == "frustrated":
        return "+5%", "+0Hz"

    if tone == "confused":
        return "-3%", "+0Hz"

    if tone == "calm":
        return "-2%", "+0Hz"

    return "+0%", "+0Hz"


# ============================================================
# EDGE TTS
# ============================================================

async def generate_tts_bytes(
    text: str,
    voice: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bytes:
    """
    Generate TTS audio completely in memory.

    No temporary audio file is created.
    """

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    audio_data = bytearray()

    async for chunk in communicate.stream():

        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])

    if not audio_data:
        raise RuntimeError(
            "Edge TTS returned no audio data."
        )

    return bytes(audio_data)


# ============================================================
# MP3 → PCM
# ============================================================

def decode_mp3_to_pcm(mp3_bytes: bytes):
    """
    Decode MP3 bytes directly in memory using miniaudio.

    Returns:
        PCM bytes
    """

    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=CHANNELS,
        sample_rate=SAMPLE_RATE,
    )

    return decoded.samples


# ============================================================
# SILENCE
# ============================================================

def create_silence(duration_ms: int) -> bytes:
    """
    Create 16-bit mono PCM silence.
    """

    number_of_samples = int(
        SAMPLE_RATE * duration_ms / 1000
    )

    number_of_bytes = (
        number_of_samples
        * CHANNELS
        * SAMPLE_WIDTH
    )

    return b"\x00" * number_of_bytes


# ============================================================
# LOAD CONVERSATION
# ============================================================

def load_conversation(path: Path):
    """
    Load conversation JSON.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# GENERATE ONE CONVERSATION
# ============================================================

async def generate_conversation(
    conversation_path: Path
):
    """
    Generate one complete conversation as one WAV file.

    All intermediate audio stays in memory.
    """

    conversation = load_conversation(
        conversation_path
    )

    conversation_id = conversation[
        "conversation_id"
    ]

    customer_tone = conversation[
        "customer_tone"
    ]

    turns = conversation["turns"]

    print()
    print("=" * 70)
    print(f"Generating audio: {conversation_id}")
    print(f"Customer tone: {customer_tone}")
    print(f"Turns: {len(turns)}")
    print("=" * 70)

    # Store the complete conversation PCM in memory.
    final_pcm = bytearray()

    for index, turn in enumerate(
        turns,
        start=1
    ):

        speaker = turn["speaker"]
        text = turn["text"]

        voice = get_voice(speaker)

        rate, pitch = get_speech_parameters(
            speaker,
            customer_tone
        )

        print(
            f"[{index:03d}/{len(turns):03d}] "
            f"{speaker}: "
            f"{text[:75]}"
        )

        # ----------------------------------------------------
        # Generate TTS directly in memory
        # ----------------------------------------------------

        mp3_bytes = await generate_tts_bytes(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        # ----------------------------------------------------
        # Decode MP3 → PCM directly in memory
        # ----------------------------------------------------

        pcm_bytes = decode_mp3_to_pcm(
            mp3_bytes
        )

        # ----------------------------------------------------
        # Append speech to final conversation
        # ----------------------------------------------------

        final_pcm.extend(pcm_bytes)

        # ----------------------------------------------------
        # Add pause between turns
        # ----------------------------------------------------

        if index < len(turns):

            final_pcm.extend(
                create_silence(PAUSE_MS)
            )

    # ========================================================
    # WRITE FINAL WAV
    # ========================================================

    AUDIO_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        AUDIO_DIR
        / f"{conversation_id}.wav"
    )

    with wave.open(
        str(output_path),
        "wb"
    ) as wav_file:

        wav_file.setnchannels(
            CHANNELS
        )

        wav_file.setsampwidth(
            SAMPLE_WIDTH
        )

        wav_file.setframerate(
            SAMPLE_RATE
        )

        wav_file.writeframes(
            bytes(final_pcm)
        )

    # Calculate duration
    bytes_per_second = (
        SAMPLE_RATE
        * CHANNELS
        * SAMPLE_WIDTH
    )

    duration_seconds = (
        len(final_pcm)
        / bytes_per_second
    )

    print()
    print("-" * 70)
    print(f"Created : {output_path}")
    print(
        f"Duration: "
        f"{duration_seconds / 60:.2f} minutes"
    )
    print(
        f"Format  : "
        f"{SAMPLE_RATE} Hz / "
        f"{CHANNELS} channel / "
        f"16-bit PCM"
    )
    print("-" * 70)


# ============================================================
# MAIN
# ============================================================

async def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print()
        print("  Generate C01:")
        print("    python generate_audio.py c01")
        print()
        print("  Generate all:")
        print("    python generate_audio.py all")
        print()

        return

    target = sys.argv[1].lower()

    # --------------------------------------------------------
    # Generate all conversations
    # --------------------------------------------------------

    if target == "all":

        conversation_files = sorted(
            CONVERSATIONS_DIR.glob("c*.json")
        )

        if not conversation_files:

            print(
                "No conversation JSON files found "
                "in the conversations folder."
            )

            return

        print()
        print(
            f"Found "
            f"{len(conversation_files)} "
            f"conversation files."
        )

        for path in conversation_files:

            try:

                await generate_conversation(
                    path
                )

            except Exception as e:

                print()
                print(
                    f"ERROR generating "
                    f"{path.name}: {e}"
                )

                print(
                    "Continuing with next conversation..."
                )

    # --------------------------------------------------------
    # Generate one conversation
    # --------------------------------------------------------

    else:

        conversation_path = (
            CONVERSATIONS_DIR
            / f"{target}.json"
        )

        if not conversation_path.exists():

            print()
            print(
                f"Conversation not found:"
                f"\n{conversation_path}"
            )

            return

        try:

            await generate_conversation(
                conversation_path
            )

        except Exception as e:

            print()
            print(
                f"ERROR: {e}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())