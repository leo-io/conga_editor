"""Rekordbox analysis dump to SongMap parser."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

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


def phrase_label(mood: int, kind: int, k1: int = 0, k2: int = 0, k3: int = 0) -> str:
    """Return the display label for a phrase or section kind in a given mood."""
    labels = MOOD_LABELS.get(mood, {})
    base = labels.get(kind, f"Kind {kind}")
    if mood == 1 and k1:
        return f"{base} {k1}"
    return base


@dataclass(frozen=True)
class Section:
    index: int
    kind: int
    label: str
    start_beat: int
    end_beat: int  # 1-based beat numbers
    start_s: float
    end_s: float
    bars: float
    start_phrase_index: int = 1
    phrase_count: int = 1
    beats: int = 0
    fill: int = 0
    beat_fill: int | None = None
    k1: int = 0
    k2: int = 0
    k3: int = 0

    @property
    def end_phrase_index(self) -> int:
        return self.start_phrase_index + self.phrase_count - 1

    @property
    def phrase_range_str(self) -> str:
        if self.phrase_count <= 1:
            return f"phrase {self.start_phrase_index}"
        return f"phrases {self.start_phrase_index}-{self.end_phrase_index}"


SongSection = Section
Phrase = Section


def group_phrases(
    phrases: list[dict],
    mood: int | None = None,
    end_beat: int | None = None,
) -> list[dict]:
    """Group contiguous phrases with matching kind into macroscopic song sections.

    Adjacent phrases in beat order are merged when:
    - kind matches,
    - the preceding section's end_beat equals the phrase's beat, and
    - for mood == 1 (High), the variant flags (k1, k2, k3) also match.

    The merged section retains the starting phrase's index as start_phrase_index,
    numbers sections monotonically from 1 to N, sums the beat and bar counts,
    and takes fill and beat_fill from the last phrase in the run.
    """
    if not phrases:
        return []

    sorted_phrases = sorted(phrases, key=lambda p: int(p.get("beat", 0)))
    total_end = (
        int(end_beat)
        if end_beat is not None
        else int(sorted_phrases[-1].get("beat", 0)) + 32
    )

    sections: list[dict] = []
    for i, p in enumerate(sorted_phrases):
        start_beat = int(p.get("beat", 0))
        next_beat = (
            int(sorted_phrases[i + 1].get("beat", 0))
            if i + 1 < len(sorted_phrases)
            else total_end
        )
        kind = int(p.get("kind", 0))
        k1 = int(p.get("k1", 0))
        k2 = int(p.get("k2", 0))
        k3 = int(p.get("k3", 0))
        fill = int(p.get("fill", 0))
        raw_bfill = p.get("beat_fill", 0)
        beat_fill = int(raw_bfill) if fill and raw_bfill else None

        can_merge = False
        if sections:
            prev = sections[-1]
            variants_match = (
                (prev.get("k1") == k1 and prev.get("k2") == k2 and prev.get("k3") == k3)
                if mood == 1
                else True
            )
            if prev["kind"] == kind and prev["end_beat"] == start_beat and variants_match:
                can_merge = True

        if can_merge:
            prev = sections[-1]
            prev["end_beat"] = next_beat
            prev["beats"] = prev["end_beat"] - prev["start_beat"]
            prev["bars"] = round(prev["beats"] / 4.0, 2)
            prev["phrase_count"] += 1
            prev["fill"] = fill
            prev["beat_fill"] = beat_fill
        else:
            beats = next_beat - start_beat
            start_phrase_index = int(p.get("index", i + 1))
            sections.append(
                {
                    "index": len(sections) + 1,
                    "start_phrase_index": start_phrase_index,
                    "kind": kind,
                    "label": phrase_label(mood or 2, kind, k1, k2, k3),
                    "start_beat": start_beat,
                    "end_beat": next_beat,
                    "beats": beats,
                    "bars": round(beats / 4.0, 2),
                    "phrase_count": 1,
                    "fill": fill,
                    "beat_fill": beat_fill,
                    "k1": k1,
                    "k2": k2,
                    "k3": k3,
                }
            )
    return sections


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
    phrases: list[Section]
    preview_bands: np.ndarray  # (1200, 3) uint8 low/mid/high
    raw_phrases: list[Section] = field(default_factory=list)
    phrase_starts_s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if len(self.phrase_starts_s) == 0 and self.phrases:
            self.phrase_starts_s = np.array([p.start_s for p in self.phrases], dtype=np.float64)

    @property
    def sections(self) -> list[Section]:
        """Macroscopic song sections (grouped phrases or raw phrases if not grouped)."""
        return self.phrases

    @property
    def has_grouped_sections(self) -> bool:
        """True if sections are grouped phrases rather than 1:1 raw Rekordbox phrases."""
        return bool(self.raw_phrases and len(self.phrases) != len(self.raw_phrases))

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

    def phrase_at(self, t: float) -> Section | None:
        """Return the Section/Phrase covering time t, or None if outside all."""
        if not self.phrases or len(self.phrase_starts_s) == 0:
            return None
        idx = int(np.searchsorted(self.phrase_starts_s, t, side="right") - 1)
        if idx < 0:
            return None
        p = self.phrases[idx]
        if p.start_s <= t < p.end_s:
            return p
        if idx == len(self.phrases) - 1 and t == p.end_s:
            return p
        return None

    def section_at(self, t: float) -> Section | None:
        """Alias for phrase_at."""
        return self.phrase_at(t)

    def snap(self, t: float, mode: str) -> float:
        """Snap time t to the nearest boundary according to mode.

        mode in {"beat", "bar", "phrase", "section", "off"}.
        """
        mode = mode.lower()
        if mode == "off":
            return float(t)
        elif mode == "beat":
            boundaries = self.beat_times_s
        elif mode == "bar":
            boundaries = self.bar_times_s
        elif mode in ("phrase", "section"):
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


def load_song(folder: Path | str) -> SongMap:
    """Load song metadata, beat grid, song structure, and preview bands from folder."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Song folder does not exist: {folder}")

    # If user pointed to an analysis subfolder, ascend to track root
    if folder.name in ("raw_analisys_data", "raw_analysis_data", "_temp", "analysis") or not (folder / "track.json").exists() and (folder.parent / "track.json").exists():
        folder = folder.parent

    # Read track.json if present
    track_meta_path = folder / "track.json"
    artist = ""
    title = ""
    slug = folder.name
    audio_path: Path | None = None
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

    # Locate track_dump.json
    dump_candidates = [
        folder / "raw_analisys_data" / "track_dump.json",
        folder / "raw_analysis_data" / "track_dump.json",
        folder / "_temp" / "track_dump.json",
        folder / "track_dump.json",
    ]
    dump_path = None
    for cand in dump_candidates:
        if cand.exists():
            dump_path = cand
            break

    if dump_path is None:
        raise FileNotFoundError(
            f"Cannot find track_dump.json in {folder} (checked root, raw_analisys_data, raw_analysis_data, _temp)"
        )

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

    def beat_to_time(beat_1based: int) -> float:
        """Map 1-based beat number to time in seconds."""
        if len(beat_times_s) == 0:
            return 0.0
        idx = beat_1based - 1
        if idx < len(beat_times_s):
            return float(beat_times_s[max(0, idx)])
        if len(beat_times_s) > 1:
            dt = float(beat_times_s[-1] - beat_times_s[-2])
            return float(beat_times_s[-1] + (idx - (len(beat_times_s) - 1)) * dt)
        return float(duration_s) if duration_s > 0 else float(beat_times_s[-1])

    # Parse phrases from EXT song_structure
    ext = anlz.get("EXT", {})
    ext_sections = ext.get("sections", {})
    ss = ext_sections.get("song_structure", {})
    mood = ss.get("mood", 2)
    end_beat_total = ss.get("end_beat", len(beat_times_s))
    raw_phrases_data = ss.get("phrases", [])

    # Parse raw Rekordbox phrases
    raw_phrases: list[Section] = []
    for i, p in enumerate(raw_phrases_data):
        start_beat = int(p["beat"])
        if i + 1 < len(raw_phrases_data):
            end_beat = int(raw_phrases_data[i + 1]["beat"])
        else:
            end_beat = int(end_beat_total)

        start_s = beat_to_time(start_beat)
        end_s = beat_to_time(end_beat)
        beats = end_beat - start_beat
        bars = float(beats / 4.0)
        k1 = int(p.get("k1", 0))
        k2 = int(p.get("k2", 0))
        k3 = int(p.get("k3", 0))
        label = phrase_label(mood, int(p["kind"]), k1, k2, k3)
        raw_phrases.append(
            Section(
                index=int(p.get("index", i + 1)),
                kind=int(p["kind"]),
                label=label,
                start_beat=start_beat,
                end_beat=end_beat,
                start_s=start_s,
                end_s=end_s,
                bars=bars,
                start_phrase_index=int(p.get("index", i + 1)),
                phrase_count=1,
                beats=beats,
                fill=int(p.get("fill", 0)),
                beat_fill=p.get("beat_fill"),
                k1=k1,
                k2=k2,
                k3=k3,
            )
        )

    # Check for new section mapping (grouped sections)
    # 1. Derived song_structure sections (already formatted with labels)
    # 2. Raw song_sections in EXT sections
    sections_source = ext.get("derived", {}).get("song_structure", {}).get("sections")
    if not sections_source:
        sections_source = ext_sections.get("song_sections")

    phrases: list[Section] = []
    if sections_source:
        for i, s in enumerate(sections_source):
            start_beat = int(s["start_beat"])
            end_beat = int(s["end_beat"])
            start_s = beat_to_time(start_beat)
            end_s = beat_to_time(end_beat)
            beats = int(s.get("beats", end_beat - start_beat))
            bars = float(s.get("bars", beats / 4.0))
            k1 = int(s.get("k1", 0))
            k2 = int(s.get("k2", 0))
            k3 = int(s.get("k3", 0))
            label = s.get("label") or phrase_label(mood, int(s["kind"]), k1, k2, k3)
            sec_idx = int(s.get("index", i + 1))
            start_phrase_idx = int(s.get("start_phrase_index", sec_idx))
            phrase_cnt = int(s.get("phrase_count", 1))

            phrases.append(
                Section(
                    index=sec_idx,
                    kind=int(s["kind"]),
                    label=label,
                    start_beat=start_beat,
                    end_beat=end_beat,
                    start_s=start_s,
                    end_s=end_s,
                    bars=bars,
                    start_phrase_index=start_phrase_idx,
                    phrase_count=phrase_cnt,
                    beats=beats,
                    fill=int(s.get("fill", 0)),
                    beat_fill=s.get("beat_fill"),
                    k1=k1,
                    k2=k2,
                    k3=k3,
                )
            )
    else:
        # Fallback to raw phrases if no grouped sections exist
        phrases = raw_phrases

    # 3-band preview bands
    preview_candidates = [
        folder / "raw_analisys_data" / "anlz.EX2.waveform_3band_preview.data.json",
        folder / "raw_analysis_data" / "anlz.EX2.waveform_3band_preview.data.json",
        folder / "_temp" / "anlz.EX2.waveform_3band_preview.data.json",
        folder / "anlz.EX2.waveform_3band_preview.data.json",
    ]
    preview_path = None
    for cand in preview_candidates:
        if cand.exists():
            preview_path = cand
            break

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
        raw_phrases=raw_phrases,
    )
