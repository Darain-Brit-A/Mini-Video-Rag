"""Small shared helpers used by the Mini Video RAG pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
CHUNK_DIR = DATA_DIR / "chunks"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"


def ensure_project_directories() -> None:
    """Create the simple local storage folders used by this one-video demo."""
    for directory in (VIDEO_DIR, TRANSCRIPT_DIR, CHUNK_DIR, VECTORSTORE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def format_timestamp(seconds: float) -> str:
    """Turn seconds into a readable HH:MM:SS timestamp."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
