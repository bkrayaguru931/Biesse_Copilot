import pyaudiowpatch as pyaudio
import wave

OUTPUT_FILE = "loopback_test.wav"
DURATION = 10

with pyaudio.PyAudio() as p:

    # Get the loopback device for your default speakers
    loopback = p.get_default_wasapi_loopback()

    print("Capturing:")
    print(loopback["name"])
    print("Device index:", loopback["index"])
    print("Sample rate:", loopback["defaultSampleRate"])

    sample_rate = int(loopback["defaultSampleRate"])
    channels = loopback["maxInputChannels"]

    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sample_rate,
        input=True,
        input_device_index=loopback["index"],
        frames_per_buffer=1024,
    )

    frames = []

    print("\nStart playing your C01.wav now...")
    print("Capturing for 10 seconds...\n")

    for _ in range(int(sample_rate / 1024 * DURATION)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(OUTPUT_FILE, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))

print(f"\nSaved: {OUTPUT_FILE}")