"""Audio extraction and timestamp-preserving Whisper transcription."""

import json
import shutil
from pathlib import Path

import ffmpeg
import imageio_ffmpeg
from faster_whisper import WhisperModel

from .utils import format_timestamp


def extract_audio(video_path: Path, audio_path: Path) -> Path:
    """Extract mono 16 kHz WAV audio, the format Whisper handles well."""
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import av

        with av.open(str(video_path)) as container:
            if not container.streams.audio:
                raise RuntimeError("This file has no audio track.")
    except RuntimeError:
        raise
    except Exception:
        pass

    ffmpeg_executable = shutil.which("ffmpeg")
    if ffmpeg_executable is None:
        try:
            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as error:
            raise RuntimeError(
                "FFmpeg is unavailable. Install FFmpeg or run `pip install -r requirements.txt`."
            ) from error
    try:
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ac=1, ar=16000, format="wav")
            .overwrite_output()
            .run(cmd=ffmpeg_executable, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as error:
        stderr_text = error.stderr.decode("utf-8", errors="ignore") if error.stderr else ""
        if "does not contain any stream" in stderr_text:
            raise RuntimeError("This file has no audio track.") from error
        raise RuntimeError(
            "FFmpeg could not extract audio. Check that FFmpeg is installed and on PATH."
        ) from error
    return audio_path


def transcribe_audio(
    audio_path: Path, model_name: str = "small"
) -> tuple[list[dict[str, object]], str]:
    """Transcribe audio into readable segments with numeric and display timestamps."""
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), vad_filter=True)
    transcript = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            transcript.append(
                {
                    "start": round(float(segment.start), 2),
                    "end": round(float(segment.end), 2),
                    "start_time": format_timestamp(segment.start),
                    "end_time": format_timestamp(segment.end),
                    "text": text,
                }
            )
    detected_language = info.language if info else "en"
    return transcript, detected_language


def save_transcript(transcript: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
