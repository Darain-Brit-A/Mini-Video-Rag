"""Convert short Whisper segments into overlapping, timestamped evidence chunks."""

import json
from pathlib import Path


def build_chunks(
    segments: list[dict[str, object]], max_words: int = 150, overlap_segments: int = 1
) -> list[dict[str, object]]:
    """Group segments without splitting an individual transcript segment."""
    chunks = []
    current = []
    current_words = 0
    chunk_id = 1

    for segment in segments:
        segment_words = len(str(segment["text"]).split())
        if current and current_words + segment_words > max_words:
            chunks.append(_make_chunk(chunk_id, current))
            chunk_id += 1
            current = current[-overlap_segments:] if overlap_segments else []
            current_words = sum(len(str(item["text"]).split()) for item in current)
        current.append(segment)
        current_words += segment_words

    if current:
        chunks.append(_make_chunk(chunk_id, current))
    return chunks


def _make_chunk(chunk_id: int, segments: list[dict[str, object]]) -> dict[str, object]:
    first, last = segments[0], segments[-1]
    return {
        "chunk_id": chunk_id,
        "start": first["start"],
        "end": last["end"],
        "start_time": first["start_time"],
        "end_time": last["end_time"],
        "text": " ".join(str(segment["text"]) for segment in segments),
    }


def save_chunks(chunks: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
