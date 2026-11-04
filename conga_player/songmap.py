"""Rekordbox analysis dump to SongMap parser."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
import warnings
import numpy as np

from conga_player.peaks import compute_fallback_preview_bands

MOOD_LABELS = {
    1: {  # High
        1: "Intro",
        2: "Up",
        3: "Down",
        4: "Chorus",
        5: "Outro",
    },
    2: {  # Mid
        1: "Intro",
        2: "Verse1",
        3: "Verse2",
        4: "Verse3",
        5: "Verse4",
        6: "Verse5",
        7: "Verse6",
        8: "Bridge",
        9: "Chorus",
        10: "Outro",
    },
    3: {  # Low
        1: "Intro",
        2: "Verse1",
        3: "Verse2",
        4: "Bridge",
        5: "Chorus",
        6: "Outro",
    },
}


def phrase_label(mood: int, kind: int) -> str:
    """Return the display label for a phrase kind in a given mood."""
    return MOOD_LABELS.get(mood, {}).get(kind, f"Kind {kind}")


@dataclass(frozen=True)
class Phrase:
    index: int
    kind: int
    label: str
    start_beat: int
    end_beat: int  # 1-based beat numbers
    start_s: float
    end_s: float
    bars: float


@dataclass
class SongMap:
    slug: str
    artist: str
    title: str
    key: str
    audio_path: Path
    duration_s: float
    sample_rate: int
    bpm: float
    beat_times_s: np.ndarray  # (N,) float64, strictly increasing
    beat_numbers: np.ndarray  # (N,) int8, 1..4
    bar_times_s: np.ndarray  # (M,) downbeats only
    bar_first_beat: np.ndarray  # (M,) index into beat_times_s
    phrases: list[Phrase]
    preview_bands: np.ndarray  # (1200, 3) uint8 low/mid/high
    grid_source: str = "rekordbox"  # "rekordbox" | "beat_this"

    def beat_index_at(self, t: float) -> int:
        """Return 0-based beat index active at time t (seconds)."""
        if len(self.beat_times_s) == 0:
            return 0
        idx = int(np.searchsorted(self.beat_times_s, t, side="right") - 1)
        return max(0, min(idx, len(self.beat_times_s) - 1))

    def bar_index_at(self, t: float) -> int:
        """Return 0-based bar index active at time t (seconds)."""
        if len(self.bar_times_s) == 0:
            return 0
        idx = int(np.searchsorted(self.bar_times_s, t, side="right") - 1)
        return max(0, min(idx, len(self.bar_times_s) - 1))

    def phrase_at(self, t: float) -> Optional[Phrase]:
        """Return the Phrase covering time t, or None if outside all phrases."""
        for p in self.phrases:
            if p.start_s <= t < p.end_s:
                return p
        if self.phrases and t == self.phrases[-1].end_s:
            return self.phrases[-1]
        return None

    def snap(self, t: float, mode: str) -> float:
        """Snap time t to the nearest boundary according to mode.

        mode in {"beat", "bar", "phrase", "off"}.
        """
        mode = mode.lower()
        if mode == "off":
            return float(t)
        elif mode == "beat":
            boundaries = self.beat_times_s
        elif mode == "bar":
            boundaries = self.bar_times_s
        elif mode == "phrase":
            if not self.phrases:
                return float(t)
            boundaries = np.array(
                [p.start_s for p in self.phrases] + [self.phrases[-1].end_s],
                dtype=np.float64,
            )
        else:
            return float(t)

        if len(boundaries) == 0:
            return float(t)

        idx = np.searchsorted(boundaries, t)
        if idx == 0:
            return float(boundaries[0])
        if idx >= len(boundaries):
            return float(boundaries[-1])

        left = boundaries[idx - 1]
        right = boundaries[idx]
        if abs(t - left) <= abs(t - right):
            return float(left)
        else:
            return float(right)


DATA_SUBDIRS = ("_temp", "raw_analisys_data", ".")


def _first_existing(folder: Path, name: str) -> Optional[Path]:
    """Return folder/<subdir>/name for the first analysis subdir that has it."""
    for sub in DATA_SUBDIRS:
        cand = folder / sub / name
        if cand.exists():
            return cand
    return None


GRID_SOURCES = ("auto", "rekordbox", "beat_this")


def find_beat_this_file(folder: Path) -> Optional[Path]:
    """Return the first *.beats file in folder/beat_this (or folder itself)."""
    for sub in ("beat_this", "."):
        found = sorted((folder / sub).glob("*.beats"))
        if found:
            return found[0]
    return None


def parse_beat_this(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a beat_this .beats file: one '<time_s> <beat_number>' per line.

    Beat number 1 marks a downbeat. Bars may have any length (no 4/4 assumption).
    """
    times: list[float] = []
    nums: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected '<time> <beat_number>', got {line!r}")
            times.append(float(parts[0]))
            nums.append(int(float(parts[1])))
    beat_times_s = np.array(times, dtype=np.float64)
    if len(beat_times_s) > 1 and not np.all(np.diff(beat_times_s) > 0):
        raise ValueError(f"{path}: beat times are not strictly increasing")
    return beat_times_s, np.array(nums, dtype=np.int8)


def _rekordbox_grid(dump: dict, duration_s: float) -> tuple[np.ndarray, np.ndarray, float, list[Phrase]]:
    """Beat grid, BPM and phrases from a rekordbox track_dump."""
    anlz = dump.get("anlz", {})
    dat_beats = anlz.get("DAT", {}).get("sections", {}).get("beat_grid", {}).get("beats", [])
    ext_beats = anlz.get("EXT", {}).get("sections", {}).get("extended_beat_grid", {}).get("beats", [])

    if len(dat_beats) == len(ext_beats) and len(dat_beats) > 0:
        times_ms = [
            b["time"] + eb.get("microseconds", 0) / 1000.0
            for b, eb in zip(dat_beats, ext_beats)
        ]
    elif len(dat_beats) > 0:
        times_ms = [b["time"] for b in dat_beats]
    else:
        times_ms = []

    beat_times_s = np.array(times_ms, dtype=np.float64) / 1000.0
    beat_numbers = np.array([b["beat_number"] for b in dat_beats], dtype=np.int8)

    bpm = float(dat_beats[0]["tempo"] / 100.0) if dat_beats else 120.0

    bar_first_beat = np.where(beat_numbers == 1)[0]
    bar_times_s = beat_times_s[bar_first_beat]

    # Verify 4/4
    if len(bar_first_beat) > 1:
        diffs = np.diff(bar_first_beat)
        if not np.all(diffs == 4):
            warnings.warn(f"Track is not strictly 4/4: non-4 bar steps found: {set(diffs)}")

    # Parse phrases from EXT song_sections (grouped sections) if available, fallback to song_structure
    ext_sections = anlz.get("EXT", {}).get("sections", {})
    ss = ext_sections.get("song_structure", {})
    mood = ss.get("mood", 2)
    end_beat_total = ss.get("end_beat", len(beat_times_s))
    raw_sections = ext_sections.get("song_sections", [])

    phrases: list[Phrase] = []
    if raw_sections:
        for i, s in enumerate(raw_sections):
            start_beat = s["start_beat"]
            end_beat = s["end_beat"]
            start_s = float(beat_times_s[start_beat - 1])
            end_s = float(beat_times_s[end_beat - 1]) if end_beat - 1 < len(beat_times_s) else float(duration_s)
            bars = float(s.get("bars", (end_beat - start_beat) / 4.0))
            label = phrase_label(mood, s["kind"])
            phrases.append(
                Phrase(
                    index=s.get("index", i + 1),
                    kind=s["kind"],
                    label=label,
                    start_beat=start_beat,
                    end_beat=end_beat,
                    start_s=start_s,
                    end_s=end_s,
                    bars=bars,
                )
            )
    else:
        raw_phrases = ss.get("phrases", [])
        for i, p in enumerate(raw_phrases):
            start_beat = p["beat"]
            if i + 1 < len(raw_phrases):
                end_beat = raw_phrases[i + 1]["beat"]
            else:
                end_beat = end_beat_total

            start_s = float(beat_times_s[start_beat - 1])
            end_s = float(beat_times_s[end_beat - 1])
            bars = float((end_beat - start_beat) / 4.0)
            label = phrase_label(mood, p["kind"])
            phrases.append(
                Phrase(
                    index=p.get("index", i + 1),
                    kind=p["kind"],
                    label=label,
                    start_beat=start_beat,
                    end_beat=end_beat,
                    start_s=start_s,
                    end_s=end_s,
                    bars=bars,
                )
            )

    return beat_times_s, beat_numbers, bpm, phrases


def load_song(folder: Path | str, grid: str = "auto") -> SongMap:
    """Load song metadata, beat grid, song structure, and preview bands from folder.

    grid selects where beats/bars/phrases come from:
      "rekordbox" — track_dump.json (beat grid + phrase analysis)
      "beat_this" — beat_this/*.beats (beats, downbeats and bars only; no phrases)
      "auto"      — rekordbox if track_dump.json exists, otherwise beat_this
    Metadata (key, title, …) and the waveform preview still come from the rekordbox
    files when they exist, whichever grid is used.
    """
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Song folder does not exist: {folder}")
    grid = grid.lower().replace("-", "_")
    if grid not in GRID_SOURCES:
        raise ValueError(f"grid must be one of {', '.join(GRID_SOURCES)}, got {grid!r}")

    # Read track.json if present
    track_meta_path = folder / "track.json"
    artist = ""
    title = ""
    slug = folder.name
    audio_path: Optional[Path] = None
    sample_rate = 44100
    duration_s = 0.0

    if track_meta_path.exists():
        with open(track_meta_path, "r", encoding="utf-8") as f:
            track_meta = json.load(f)
        artist = track_meta.get("artist") or track_meta.get("source_tags", {}).get("artist", "")
        title = track_meta.get("title") or track_meta.get("source_tags", {}).get("title", "")
        slug = track_meta.get("title_slug") or folder.name
        files = track_meta.get("files", [])
        if files:
            first_file = files[0]
            for candidate in files:
                if candidate.get("role") == "main":
                    first_file = candidate
                    break
            rel_audio = first_file.get("path", "")
            if rel_audio:
                audio_path = folder / rel_audio
            sample_rate = first_file.get("sample_rate", 44100)
            duration_s = float(first_file.get("duration", 0.0))

    if audio_path is None or not audio_path.exists():
        for ext in ("*.flac", "*.wav", "*.mp3", "*.m4a"):
            found = list(folder.glob(ext))
            if found:
                audio_path = found[0]
                break

    if audio_path is None:
        raise FileNotFoundError(f"No audio file found in {folder}")

    # Locate track_dump.json (required for the rekordbox grid, optional metadata otherwise)
    dump_path = _first_existing(folder, "track_dump.json")
    beats_path = find_beat_this_file(folder)
    if grid == "auto":
        grid = "rekordbox" if dump_path is not None else "beat_this"
    if grid == "rekordbox" and dump_path is None:
        raise FileNotFoundError(
            f"Cannot find track_dump.json in {folder}, {folder / '_temp'} or {folder / 'raw_analisys_data'}"
        )
    if grid == "beat_this" and beats_path is None:
        raise FileNotFoundError(f"Cannot find a beat_this *.beats file in {folder / 'beat_this'} or {folder}")

    dump: dict = {}
    if dump_path is not None:
        with open(dump_path, "r", encoding="utf-8") as f:
            dump = json.load(f)

    trk_info = dump.get("track", {})
    key = trk_info.get("key", "")
    if not artist:
        artist = trk_info.get("artist", "")
    if not title:
        title = trk_info.get("title", "")
    if duration_s <= 0 and trk_info.get("length"):
        duration_s = float(trk_info["length"])
    if trk_info.get("sample_rate"):
        sample_rate = int(trk_info["sample_rate"])

    if grid == "rekordbox":
        beat_times_s, beat_numbers, bpm, phrases = _rekordbox_grid(dump, duration_s)
    else:
        beat_times_s, beat_numbers = parse_beat_this(beats_path)
        phrases = []
        ibi = np.diff(beat_times_s)
        bpm = float(60.0 / np.median(ibi)) if len(ibi) else 120.0
    bar_first_beat = np.where(beat_numbers == 1)[0]
    bar_times_s = beat_times_s[bar_first_beat]

    # 3-band preview bands
    preview_path = _first_existing(folder, "anlz.EX2.waveform_3band_preview.data.json")

    if preview_path is not None:
        with open(preview_path, "r", encoding="utf-8") as f:
            preview_json = json.load(f)
        preview_bands = np.array(
            [[item.get("low", 0), item.get("mid", 0), item.get("high", 0)] for item in preview_json],
            dtype=np.uint8,
        )
    else:
        preview_bands = compute_fallback_preview_bands(audio_path, sample_rate, 1200)

    if duration_s <= 0 and len(beat_times_s) > 0:
        duration_s = float(beat_times_s[-1])

    return SongMap(
        slug=slug,
        artist=artist,
        title=title,
        key=key,
        audio_path=audio_path,
        duration_s=duration_s,
        sample_rate=sample_rate,
        bpm=bpm,
        beat_times_s=beat_times_s,
        beat_numbers=beat_numbers,
        bar_times_s=bar_times_s,
        bar_first_beat=bar_first_beat,
        phrases=phrases,
        preview_bands=preview_bands,
        grid_source=grid,
    )
