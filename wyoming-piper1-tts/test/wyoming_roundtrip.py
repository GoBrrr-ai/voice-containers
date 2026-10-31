#!/usr/bin/env python
"""Wyoming TTS/STT round-trip helper. Runs INSIDE the container (wyoming lib).

Usage (via docker exec):
  ... python - tts <wav-out> "<text>"
  ... python - stt <wav-in>
"""

import asyncio
import sys
import wave

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

PIPER_PORT = 10200
WHISPER_PORT = 10300


async def do_tts(text: str, out: str) -> None:
    audio = bytearray()
    rate = 22050
    async with AsyncTcpClient("localhost", PIPER_PORT) as client:
        await client.write_event(Synthesize(text=text).event())
        while True:
            ev = await client.read_event()
            if ev is None:
                break
            if AudioStart.is_type(ev.type):
                rate = AudioStart.from_event(ev).rate
            elif AudioChunk.is_type(ev.type):
                audio += AudioChunk.from_event(ev).audio
            elif AudioStop.is_type(ev.type):
                break
    if not audio:
        raise SystemExit("no audio received from piper")
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(audio))
    print(f"TTS OK: {len(audio)} bytes @ {rate} Hz -> {out}")


async def do_stt(wav_path: str) -> str:
    with wave.open(wav_path, "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        data = w.readframes(w.getnframes())
    chunk_size = 1024 * width * channels
    async with AsyncTcpClient("localhost", WHISPER_PORT) as client:
        await client.write_event(Transcribe().event())
        await client.write_event(
            AudioStart(rate=rate, width=width, channels=channels).event()
        )
        for i in range(0, len(data), chunk_size):
            await client.write_event(
                AudioChunk(
                    rate=rate,
                    width=width,
                    channels=channels,
                    audio=data[i : i + chunk_size],
                ).event()
            )
        await client.write_event(AudioStop().event())
        while True:
            ev = await client.read_event()
            if ev is None:
                raise SystemExit("connection closed before Transcript")
            if Transcript.is_type(ev.type):
                return Transcript.from_event(ev).text


def main() -> None:
    mode = sys.argv[1]
    if mode == "tts":
        asyncio.run(do_tts(sys.argv[3], sys.argv[2]))
    elif mode == "stt":
        print(asyncio.run(do_stt(sys.argv[2])))
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
