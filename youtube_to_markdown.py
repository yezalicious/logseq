#!/usr/bin/env python3
"""
youtube_to_markdown.py

Convert a YouTube video (or Short) into a detailed markdown screening script
by combining the VTT transcript, video frames, and description via Claude.

Usage:
    python3 youtube_to_markdown.py <youtube_url> [output.md]

Requires:
    - ANTHROPIC_API_KEY environment variable
    - yt-dlp  (pip install yt-dlp)
    - anthropic (pip install anthropic)
    - ffmpeg  (apt install ffmpeg / brew install ffmpeg)
"""

import sys
import os
import json
import re
import base64
import tempfile
import subprocess
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def die(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def check_dependencies() -> None:
    missing = [t for t in ("ffmpeg", "yt-dlp") if run(["which", t]).returncode != 0]
    if missing:
        die(f"Missing tools: {', '.join(missing)}. Install with: pip install yt-dlp && apt install ffmpeg")


# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

_YTDLP_BASE = ["yt-dlp", "--no-check-certificate"]


def fetch_info(url: str) -> dict:
    print("Fetching video metadata…")
    result = run([*_YTDLP_BASE, "--dump-json", url])
    if result.returncode != 0:
        die(f"yt-dlp metadata failed:\n{result.stderr}")
    return json.loads(result.stdout)


def download_subtitles(url: str, output_dir: Path) -> None:
    print("Downloading subtitles…")
    run([
        *_YTDLP_BASE,
        "--write-auto-sub", "--write-sub",
        "--sub-langs", "en.*",
        "--sub-format", "vtt",
        "--skip-download",
        "-o", str(output_dir / "video"),
        url,
    ])


def download_video(url: str, output_dir: Path) -> Path | None:
    print("Downloading video…")
    result = run([
        *_YTDLP_BASE,
        "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "-o", str(output_dir / "video.%(ext)s"),
        "--merge-output-format", "mp4",
        url,
    ])
    if result.returncode != 0:
        print(f"  Warning: video download failed — frames will be skipped.\n  {result.stderr[:200]}")
        return None
    videos = [f for f in output_dir.glob("video.*") if f.suffix not in {".vtt", ".json", ".part"}]
    return videos[0] if videos else None


# ──────────────────────────────────────────────────────────────────────────────
# VTT parsing
# ──────────────────────────────────────────────────────────────────────────────

_TIME_RE = re.compile(
    r"(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})"
)
_TAG_RE = re.compile(r"<[^>]+>")


def _normalise_ts(ts: str) -> str:
    """Ensure HH:MM:SS.mmm format."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 2:
        ts = f"00:{ts}"
    return ts


def parse_vtt(content: str) -> list[dict]:
    segments: list[dict] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        m = _TIME_RE.search(lines[i])
        if m:
            start = _normalise_ts(m.group(1))
            end = _normalise_ts(m.group(2))
            texts: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip():
                clean = _TAG_RE.sub("", lines[i]).strip()
                if clean:
                    texts.append(clean)
                i += 1
            text = " ".join(texts)
            if text:
                segments.append({"start": start, "end": end, "text": text})
        else:
            i += 1

    # Deduplicate consecutive identical lines (common in auto-subs)
    deduped: list[dict] = []
    for seg in segments:
        if not deduped or deduped[-1]["text"] != seg["text"]:
            deduped.append(seg)
    return deduped


def find_vtt(output_dir: Path) -> list[dict]:
    vtt_files = sorted(output_dir.glob("*.vtt"))
    if not vtt_files:
        print("  No subtitle file found — transcript will be empty.")
        return []
    vtt_path = vtt_files[0]
    print(f"  Found subtitle: {vtt_path.name}")
    content = vtt_path.read_text(encoding="utf-8", errors="replace")
    segments = parse_vtt(content)
    print(f"  Parsed {len(segments)} transcript segments.")
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: Path, output_dir: Path, duration: int) -> tuple[list[Path], float]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Adaptive interval: aim for 20–30 frames regardless of video length
    target_frames = 25
    interval = max(1.0, duration / target_frames)
    # Shorts are ≤ 60 s; cap interval so we always get decent coverage
    if duration <= 60:
        interval = max(1.5, duration / 20)

    print(f"Extracting frames every {interval:.1f}s from {duration}s video…")
    result = run([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps=1/{interval}",
        "-q:v", "3",
        str(frames_dir / "frame_%04d.jpg"),
        "-y",
    ])
    if result.returncode != 0:
        print(f"  Warning: ffmpeg frame extraction failed.\n  {result.stderr[-300:]}")
        return [], interval

    frames = sorted(frames_dir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames.")
    return frames, interval


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ──────────────────────────────────────────────────────────────────────────────
# Claude analysis
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(info: dict, segments: list[dict], frames: list[Path], interval: float) -> list:
    title = info.get("title", "Unknown Title")
    creator = info.get("uploader", "Unknown")
    duration = info.get("duration", 0)
    url = info.get("webpage_url", "")
    description = (info.get("description") or "").strip() or "(no description)"

    raw_date = info.get("upload_date", "")
    upload_date = (
        f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else raw_date
    )

    transcript_block = (
        "\n".join(f"[{s['start']} → {s['end']}] {s['text']}" for s in segments)
        or "(no transcript available)"
    )

    content: list = [
        {
            "type": "text",
            "text": f"""You are a professional video analyst creating a thorough markdown screening script.

VIDEO METADATA
- Title: {title}
- Creator: {creator}
- Date: {upload_date}
- Duration: {duration}s
- URL: {url}

DESCRIPTION
{description}

TRANSCRIPT (timestamped)
{transcript_block}

FRAMES
Each frame below is labelled with its approximate timestamp (extracted every {interval:.1f}s).
""",
        }
    ]

    # Cap at 30 frames to stay within token budget
    step = max(1, len(frames) // 30)
    selected = frames[::step][:30]

    for idx, frame_path in enumerate(selected):
        ts_sec = idx * step * interval
        mm = int(ts_sec // 60)
        ss = int(ts_sec % 60)
        content.append({"type": "text", "text": f"\n[Frame {mm:02d}:{ss:02d}]"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image(frame_path),
                },
            }
        )

    content.append(
        {
            "type": "text",
            "text": """

Using everything above — metadata, description, transcript, and frames — produce a complete markdown screening script so that a reader who has never seen the video can fully understand and experience it.

Structure your output exactly like this:

# [Video Title]

**Creator:** …
**Published:** …
**Duration:** …
**Source:** [URL](URL)

---

## Overview
(2–4 sentence summary of what the video is and what it covers)

---

## Description
(Reproduce the original video description verbatim, formatted nicely)

---

## Full Transcript
(Clean, readable transcript — remove repetition and filler, keep it flowing)

---

## Scene-by-Scene Breakdown

### [00:00 – 00:XX] Scene title
**On screen:** (Describe exactly what is visually happening — setting, people, text overlays, graphics, cuts)
**Audio/Narration:** (What is being said or heard)
**Notes:** (Any notable editing choices, music, sound effects, pacing)

(Repeat for every distinct scene or beat in the video)

---

## Key Takeaways
- …

---

Output ONLY the markdown above. No preamble, no explanation.""",
        }
    )

    return content


def generate_script(info: dict, segments: list[dict], frames: list[Path], interval: float, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    content = build_prompt(info, segments, frames, interval)

    print("Sending to Claude for analysis…")
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        die("ANTHROPIC_API_KEY environment variable is not set.")

    check_dependencies()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        info = fetch_info(url)
        title = info.get("title", "video")
        duration = int(info.get("duration") or 0)

        download_subtitles(url, tmp_dir)
        video_path = download_video(url, tmp_dir)

        segments = find_vtt(tmp_dir)

        frames: list[Path] = []
        interval: float = 3.0
        if video_path and video_path.exists():
            frames, interval = extract_frames(video_path, tmp_dir, duration or 60)

        script = generate_script(info, segments, frames, interval, api_key)

        # Determine output path
        if output_arg:
            out_path = Path(output_arg)
        else:
            safe = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
            out_path = Path(f"{safe}.md")

        out_path.write_text(script, encoding="utf-8")
        print(f"\nDone!  Screening script saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
