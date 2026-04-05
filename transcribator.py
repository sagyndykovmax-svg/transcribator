"""
Transcribator — audio transcription via Yandex SpeechKit.

Usage:
    python transcribator.py <audio_file> [--lang ru-RU] [--output result.txt]

The audio file is split into 25-second chunks, each sent to the
Yandex SpeechKit synchronous recognition API, then the results are combined.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SPEECHKIT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
CHUNK_SECONDS = 25
FFMPEG = "ffmpeg"


def find_ffmpeg() -> str:
    """Return ffmpeg path, searching common locations on Windows."""
    for candidate in ["ffmpeg", r"C:\ffmpeg\bin\ffmpeg.exe"]:
        try:
            subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                check=True,
            )
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    # Try to find via where
    result = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip().splitlines()[0]
    raise RuntimeError(
        "ffmpeg not found. Please install ffmpeg and add it to PATH."
    )


def convert_to_pcm(ffmpeg: str, input_path: str, output_path: str) -> None:
    """Convert audio to raw PCM 16kHz mono — universally supported by SpeechKit."""
    cmd = [
        ffmpeg, "-y", "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr}")


def get_duration(ffmpeg: str, path: str) -> float:
    """Return audio duration in seconds using the original (non-PCM) file."""
    cmd = [ffmpeg, "-i", path, "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stderr.splitlines():
        if "Duration" in line:
            dur_str = line.strip().split("Duration:")[1].split(",")[0].strip()
            h, m, s = dur_str.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError("Could not determine audio duration.")


def split_pcm(pcm_path: str, chunk_dir: str, chunk_seconds: int) -> list[str]:
    """Split raw PCM (16kHz, 16-bit mono) into chunks by byte offset."""
    bytes_per_second = 16000 * 2  # 16kHz * 2 bytes per sample
    chunk_size = chunk_seconds * bytes_per_second
    chunks = []
    with open(pcm_path, "rb") as f:
        idx = 0
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.pcm")
            with open(chunk_path, "wb") as cf:
                cf.write(data)
            chunks.append(chunk_path)
            idx += 1
    return chunks


def transcribe_chunk(api_key: str, chunk_path: str, lang: str) -> str:
    """Send a single audio chunk to Yandex SpeechKit and return text."""
    with open(chunk_path, "rb") as f:
        data = f.read()

    params = {
        "lang": lang,
        "format": "lpcm",
        "sampleRateHertz": "16000",
    }
    headers = {"Authorization": f"Api-Key {api_key}"}

    for attempt in range(3):
        resp = requests.post(
            SPEECHKIT_URL,
            params=params,
            headers=headers,
            data=data,
            timeout=30,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("result", "")
        if resp.status_code == 429:
            # Rate limit — wait and retry
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(
            f"SpeechKit error {resp.status_code}: {resp.text}"
        )
    raise RuntimeError("SpeechKit: too many retries (rate limited).")


def transcribe(audio_path: str, lang: str, api_key: str, verbose: bool = True) -> str:
    ffmpeg = find_ffmpeg()

    with tempfile.TemporaryDirectory() as tmp:
        if verbose:
            print(f"Converting {audio_path} to PCM...")
        pcm_path = os.path.join(tmp, "audio.pcm")
        convert_to_pcm(ffmpeg, audio_path, pcm_path)

        if verbose:
            duration = get_duration(ffmpeg, audio_path)
            print(f"Duration: {duration:.1f}s — splitting into {CHUNK_SECONDS}s chunks...")

        chunks = split_pcm(pcm_path, tmp, CHUNK_SECONDS)
        total = len(chunks)

        if verbose:
            print(f"Transcribing {total} chunk(s)...")

        parts = []
        for i, chunk in enumerate(chunks, 1):
            if verbose:
                print(f"  [{i}/{total}] chunk {i}...", end=" ", flush=True)
            text = transcribe_chunk(api_key, chunk, lang)
            if verbose:
                print("ok")
            parts.append(text)

    return " ".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio via Yandex SpeechKit")
    parser.add_argument("audio", help="Path to audio file (m4a, mp3, wav, ogg, ...)")
    parser.add_argument("--lang", default="ru-RU", help="Language code (default: ru-RU)")
    parser.add_argument("--output", help="Output text file (default: <audio_name>.txt)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    api_key = os.getenv("YANDEX_API_KEY")
    if not api_key:
        print("Error: YANDEX_API_KEY not set. Add it to .env file.", file=sys.stderr)
        sys.exit(1)

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(Path(audio_path).with_suffix(".txt"))

    text = transcribe(audio_path, args.lang, api_key, verbose=not args.quiet)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    if not args.quiet:
        print(f"\nSaved to: {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
