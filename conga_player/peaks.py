"""Waveform peaks and preview bands computation for conga_player."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def zoom_peaks(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    columns: int,
) -> np.ndarray:
    """Compute mono-mixed max(abs()) per column over [start_s, end_s].

    Args:
        audio: Shape (channels, frames) or (frames,) float32 array.
        sr: Sample rate.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        columns: Number of peak columns to produce.

    Returns:
        np.ndarray: Shape (columns,) float32 array in [0.0, 1.0].
    """
    if columns <= 0:
        return np.zeros(0, dtype=np.float32)

    start_frame = max(0, int(round(start_s * sr)))
    total_frames = audio.shape[-1] if audio.ndim > 0 else 0
    end_frame = min(total_frames, int(round(end_s * sr)))

    if end_frame <= start_frame or total_frames == 0:
        return np.zeros(columns, dtype=np.float32)

    if audio.ndim == 2:
        slice_audio = audio[:, start_frame:end_frame]
        mono = np.max(np.abs(slice_audio), axis=0)
    else:
        mono = np.abs(audio[start_frame:end_frame])

    n_samples = len(mono)
    if n_samples == 0:
        return np.zeros(columns, dtype=np.float32)

    if n_samples >= columns:
        edges = np.linspace(0, n_samples, columns + 1, dtype=np.intp)
        return np.maximum.reduceat(mono, edges[:-1]).astype(np.float32)

    peaks = np.zeros(columns, dtype=np.float32)
    peaks[:n_samples] = mono
    return peaks


def compute_fallback_preview_bands(
    audio_or_path: Path | str | np.ndarray | None,
    sr: int = 44100,
    columns: int = 1200,
) -> np.ndarray:
    """Generate fallback 3-band preview data (shape (columns, 3) uint8) when JSON is missing."""
    if audio_or_path is None or columns <= 0:
        return np.zeros((columns, 3), dtype=np.uint8)

    audio = None
    if isinstance(audio_or_path, (str, Path)):
        p = Path(audio_or_path)
        if p.exists():
            try:
                from pedalboard.io import AudioFile
                with AudioFile(str(p)) as f:
                    audio = f.read(f.frames)
                    sr = f.samplerate
            except Exception:
                audio = None
    elif isinstance(audio_or_path, np.ndarray):
        audio = audio_or_path

    if audio is None or audio.size == 0:
        return np.zeros((columns, 3), dtype=np.uint8)

    # Simple 3-band energy approximation from audio
    total_frames = audio.shape[-1]
    duration_s = total_frames / float(sr)
    col_peaks = zoom_peaks(audio, sr, 0.0, duration_s, columns)
    # Scale to ~0..50 uint8 typical of rekordbox EX2
    val = np.clip(col_peaks * 50.0, 0, 50).astype(np.uint8)
    bands = np.zeros((columns, 3), dtype=np.uint8)
    bands[:, 0] = (val * 0.7).astype(np.uint8)  # low
    bands[:, 1] = (val * 0.9).astype(np.uint8)  # mid
    bands[:, 2] = val                           # high
    return bands
