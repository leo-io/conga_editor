"""live.toml: a human-readable, two-way synced description of what the player is doing.

This is the only module that knows the file format. Bare numbers are BEATS (1-based,
inclusive end); ``{ bars = [a, b] }``, ``{ phrase = n }`` and time strings are also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Optional

import numpy as np
import tomlkit

from conga_player.songmap import SongMap

MODES = ("sequence", "loop", "free")
MAX_REPEAT = 999
_SEGMENT_KEYS = {"beats", "bars", "phrase", "from", "to", "repeat"}


# --------------------------------------------------------------------------- time helpers


def parse_time_val(val: Any) -> float:
    """Convert a time string ('2m40s', '0m12.42s', '02:40', '1:02:40', '160s') into seconds."""
    time_str = str(val).strip().lower()
    if not time_str:
        raise ValueError("empty time value")

    if "m" in time_str or "s" in time_str:
        m_match = re.search(r"(\d+(?:\.\d+)?)m", time_str)
        s_match = re.search(r"(\d+(?:\.\d+)?)s", time_str)
        if not m_match and not s_match:
            raise ValueError(f"invalid time '{val}'")
        minutes = float(m_match.group(1)) if m_match else 0.0
        seconds = float(s_match.group(1)) if s_match else 0.0
        return minutes * 60.0 + seconds

    if ":" in time_str:
        parts = [float(p) for p in time_str.split(":")]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        raise ValueError(f"invalid time '{val}'")

    return float(time_str)


def format_time(t: float, decimals: int = 2) -> str:
    """Format seconds as m:ss.ss."""
    mins = int(t // 60)
    secs = t - mins * 60
    width = 3 + decimals
    return f"{mins}:{secs:0{width}.{decimals}f}"


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    repeat: int = 1
    label: str = ""


@dataclass
class LiveState:
    mode: str = "free"
    playing: bool = False
    volume_db: float = 0.0
    loop: Optional[Segment] = None
    segments: list[Segment] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error_lines: list[Optional[int]] = field(default_factory=list)


@dataclass
class LiveParseError:
    """The text is not valid TOML. Nothing is applied or written."""

    message: str
    line: Optional[int] = None
    col: Optional[int] = None

    def __str__(self) -> str:
        where = f"line {self.line}: " if self.line else ""
        return f"{where}{self.message}"


class SegmentError(ValueError):
    pass


def expand_regions(segments: list[Segment]) -> list[tuple[float, float]]:
    """Expand segments by repeat into the flat region list the engine plays."""
    out: list[tuple[float, float]] = []
    for seg in segments:
        out.extend([(seg.start_s, seg.end_s)] * max(1, seg.repeat))
    return out


def region_to_segment_map(segments: list[Segment]) -> list[int]:
    """For each expanded region, the index of the segment it came from."""
    out: list[int] = []
    for i, seg in enumerate(segments):
        out.extend([i] * max(1, seg.repeat))
    return out


# --------------------------------------------------------------------------- resolving


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _endpoint(v: Any, times: np.ndarray, kind: str, is_end: bool, song_map: SongMap, where: str) -> float:
    """Resolve one endpoint: an integer is a beat/bar number, a string is a time."""
    if _is_int(v):
        n = len(times)
        if not 1 <= v <= n:
            raise SegmentError(f"{where}: {kind} {v} out of range (1–{n})")
        if is_end:
            return float(times[v]) if v < n else float(song_map.duration_s)
        return float(times[v - 1])
    if isinstance(v, str):
        try:
            t = parse_time_val(v)
        except ValueError:
            raise SegmentError(f"{where}: invalid time '{v}'") from None
        if not 0.0 <= t <= song_map.duration_s + 1e-6:
            raise SegmentError(f"{where}: time '{v}' out of range (0–{format_time(song_map.duration_s)})")
        return t
    raise SegmentError(f"{where}: expected a {kind} number or a time string, got {v!r}")


def _pair(v: Any, where: str, what: str) -> tuple[Any, Any]:
    if not isinstance(v, list) or len(v) != 2:
        raise SegmentError(f"{where}: {what} needs two values [start, end]")
    return v[0], v[1]


def resolve_segment(spec: Any, song_map: SongMap, where: str) -> Segment:
    """Turn a 2-item array or an inline table into a Segment."""
    repeat = 1
    beats = song_map.beat_times_s

    if isinstance(spec, list):
        a, b = _pair(spec, where, "a segment")
        start = _endpoint(a, beats, "beat", False, song_map, where)
        end = _endpoint(b, beats, "beat", True, song_map, where)
        if _is_int(a) and _is_int(b):
            label = f"beats {a}–{b}"
        else:
            label = f"{format_time(start)}–{format_time(end)}"
    elif isinstance(spec, dict):
        unknown = sorted(set(spec) - _SEGMENT_KEYS)
        if unknown:
            raise SegmentError(f"{where}: unknown key '{unknown[0]}'")
        if "repeat" in spec:
            repeat = spec["repeat"]
            if not _is_int(repeat) or not 1 <= repeat <= MAX_REPEAT:
                raise SegmentError(f"{where}: repeat must be an integer 1–{MAX_REPEAT}")
        selectors = [k for k in ("beats", "bars", "phrase", "from") if k in spec]
        if len(selectors) != 1:
            raise SegmentError(f"{where}: use exactly one of beats / bars / phrase / from+to")
        sel = selectors[0]
        if sel == "phrase":
            n = spec["phrase"]
            phrases = song_map.phrases
            if not _is_int(n) or not 1 <= n <= len(phrases):
                raise SegmentError(f"{where}: phrase {n!r} out of range (1–{len(phrases)})")
            p = phrases[n - 1]
            start, end = p.start_s, p.end_s
            label = f"phrase {n} {p.label}"
        elif sel == "bars":
            a, b = _pair(spec["bars"], where, "bars")
            bars = song_map.bar_times_s
            start = _endpoint(a, bars, "bar", False, song_map, where)
            end = _endpoint(b, bars, "bar", True, song_map, where)
            label = f"bars {a}–{b}" if _is_int(a) and _is_int(b) else f"{format_time(start)}–{format_time(end)}"
        elif sel == "beats":
            a, b = _pair(spec["beats"], where, "beats")
            start = _endpoint(a, beats, "beat", False, song_map, where)
            end = _endpoint(b, beats, "beat", True, song_map, where)
            label = f"beats {a}–{b}" if _is_int(a) and _is_int(b) else f"{format_time(start)}–{format_time(end)}"
        else:  # from / to
            if "to" not in spec:
                raise SegmentError(f"{where}: 'from' needs a matching 'to'")
            start = _endpoint(spec["from"], beats, "beat", False, song_map, where)
            end = _endpoint(spec["to"], beats, "beat", True, song_map, where)
            label = f"{format_time(start)}–{format_time(end)}"
        if sel != "from" and "to" in spec:
            raise SegmentError(f"{where}: 'to' only goes with 'from'")
    else:
        raise SegmentError(f"{where}: expected [start, end] or an inline table")

    if start >= end:
        raise SegmentError(f"{where}: start must be before end ({format_time(start, 3)} >= {format_time(end, 3)})")
    return Segment(start_s=start, end_s=end, repeat=int(repeat), label=label)


def _parse_doc(text: str) -> Any:
    """tomlkit.parse, with every failure converted to a LiveParseError."""
    try:
        return tomlkit.parse(text)
    except Exception as exc:  # TOMLKitError and the odd ValueError/TypeError
        line = getattr(exc, "line", None)
        col = getattr(exc, "col", None)
        msg = str(exc)
        msg = re.sub(r"\s+at line \d+ col \d+$", "", msg)
        raise LiveSyntaxError(LiveParseError(msg, line, col)) from None


class LiveSyntaxError(Exception):
    def __init__(self, error: LiveParseError) -> None:
        super().__init__(str(error))
        self.error = error


def parse_live(text: str, song_map: SongMap) -> LiveState:
    """Parse and resolve live.toml. Raises LiveSyntaxError (carrying a LiveParseError) on bad TOML.

    Field errors are collected in ``state.errors`` and the offending item is skipped.
    """
    doc = _parse_doc(text)
    data = doc.unwrap()
    state = LiveState()

    def err(msg: str, line: Optional[int]) -> None:
        state.errors.append(msg)
        state.error_lines.append(line)

    mode = data.get("mode", "free")
    if mode in MODES:
        state.mode = mode
    else:
        err(f"mode: expected one of {', '.join(MODES)}, got {mode!r}", _key_line(text, "mode"))

    playing = data.get("playing", False)
    if isinstance(playing, bool):
        state.playing = playing
    else:
        err("playing: expected true or false", _key_line(text, "playing"))

    vol = data.get("volume_db", 0.0)
    if _is_int(vol) or isinstance(vol, float):
        state.volume_db = float(vol)
    else:
        err("volume_db: expected a number", _key_line(text, "volume_db"))

    if "loop" in data:
        try:
            seg = resolve_segment(data["loop"], song_map, "loop")
            state.loop = Segment(seg.start_s, seg.end_s, 1, seg.label)
        except SegmentError as exc:
            err(str(exc), _key_line(text, "loop"))

    segs = data.get("segments", [])
    if not isinstance(segs, list):
        err("segments: expected a list", _key_line(text, "segments"))
        segs = []
    for i, spec in enumerate(segs):
        try:
            state.segments.append(resolve_segment(spec, song_map, f"segments[{i}]"))
        except SegmentError as exc:
            err(str(exc), _element_line(text, "segments", i) or _key_line(text, "segments"))
    return state


def _key_line(text: str, key: str) -> Optional[int]:
    m = re.search(rf"^{re.escape(key)}\s*=", text, re.M)
    return text.count("\n", 0, m.start()) + 1 if m else None


def _element_line(text: str, key: str, idx: int) -> Optional[int]:
    """1-based line where element ``idx`` of the top-level array ``key`` starts."""
    m = re.search(rf"^{re.escape(key)}\s*=\s*\[", text, re.M)
    if not m:
        return None
    i, depth, n, expecting = m.end(), 1, -1, True
    line = text.count("\n", 0, i) + 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\n":
            line += 1
        elif c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif not c.isspace():
            if c == "," and depth == 1:
                expecting = True
            elif c == "]" and depth == 1:
                break
            else:
                if depth == 1 and expecting:
                    n += 1
                    expecting = False
                    if n == idx:
                        return line
                if c in "\"'":
                    j = i + 1
                    while j < len(text) and text[j] not in (c, "\n"):
                        j += 2 if (c == '"' and text[j] == "\\") else 1
                    i = j
                elif c in "[{":
                    depth += 1
                elif c in "]}":
                    depth -= 1
        i += 1
    return None


# --------------------------------------------------------------------------- starter file


def starter_text(song_map: SongMap) -> str:
    """The generated file created when live.toml is missing."""
    lines = [
        "# conga_player live session — edit & save, the player follows.",
        "# Units: bare numbers = BEATS (1-based, inclusive): [65, 96] = beat 65 through 96.",
        '#        { bars = [17, 24] }  { phrase = 8 }  ["2m40s", "2m50s"] / "02:40" / "160s"',
        "#        Any segment may add  repeat = N  (default 1).",
        "",
        'mode = "sequence"  # "sequence" | "loop" | "free"',
        "playing = false",
        "volume_db = 0.0",
        "# loop = { bars = [17, 24] }        # used when mode = \"loop\"; UI loop actions write here",
        "",
        "segments = [                       # used when mode = \"sequence\"; plays in order, wraps",
    ]
    for i, p in enumerate(song_map.phrases, start=1):
        lines.append(f"    {{ phrase = {i} }},  # {p.label}")
    if not song_map.phrases:
        lines.append(
            f"    # no phrases in the {song_map.grid_source} beat map — use bars, e.g. {{ bars = [1, 8] }}"
            f"  ({len(song_map.bar_times_s)} bars)"
        )
    lines.extend([
        "]",
        "",
        "# ── Map (generated, for reference) ─────────────────────────────",
    ])
    for i, p in enumerate(song_map.phrases, start=1):
        first_bar = song_map.bar_index_at(p.start_s) + 1
        last_bar = song_map.bar_index_at(max(p.start_s, p.end_s - 1e-6)) + 1
        bars_count = int(round(p.bars)) if round(p.bars, 2).is_integer() else p.bars
        bar_str = f"{bars_count} bars" if bars_count != 1 else "1 bar"
        lines.append(
            f"# phrase {i:2d}  {p.label:<8} beats {p.start_beat:4d}–{p.end_beat - 1:<4d}  "
            f"bars {first_bar:3d}–{last_bar:<3d}  ({bar_str})"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- editing


def _render(v: Any) -> str:
    """Render a Python value as readable TOML (inline tables as '{ k = v }')."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_render(x)}" for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_render(x) for x in v) + "]"
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    return repr(v)


def _to_toml(v: Any) -> Any:
    return tomlkit.parse(f"x = {_render(v)}")["x"]

_ANCHOR = re.compile(r"^(?:volume_db|playing|mode)\s*=.*$", re.M)


def edit_text(text: str, **changes: Any) -> str:
    """Apply UI changes to ``text`` in place, keeping comments, ordering and the phrase table.

    ``None`` removes a key. Raises LiveSyntaxError if ``text`` is not valid TOML.
    """
    doc = _parse_doc(text)
    missing: list[tuple[str, Any]] = []
    for key, value in changes.items():
        if value is None:
            if key in doc:
                del doc[key]
        elif key in doc:
            doc[key] = _to_toml(value)
        else:
            missing.append((key, value))
    out = doc.as_string()
    for key, value in missing:
        new_line = f"{key} = {_render(value)}"
        anchors = list(_ANCHOR.finditer(out))
        if anchors:
            end = anchors[-1].end()
            out = out[:end] + "\n" + new_line + out[end:]
        else:
            out = new_line + "\n" + out
    return out


def loop_spec_for_phrase(song_map: SongMap, phrase) -> dict:
    return {"phrase": song_map.phrases.index(phrase) + 1}


def loop_spec_for_bars(first_bar: int, last_bar: int) -> dict:
    """1-based inclusive bar numbers."""
    return {"bars": [first_bar, last_bar]}


def loop_spec_for_range(song_map: SongMap, start_s: float, end_s: float, snap_mode: str) -> Any:
    """Most readable spec for a dragged range: beats when snapped, time strings when not."""
    if start_s > end_s:
        start_s, end_s = end_s, start_s
    beats = song_map.beat_times_s
    if snap_mode != "off" and len(beats) > 0:
        a = int(np.argmin(np.abs(beats - start_s))) + 1
        if end_s >= song_map.duration_s - 1e-6:
            b = len(beats)
        else:
            b = int(np.argmin(np.abs(beats - end_s)))
        if 1 <= a < b + 1 and b <= len(beats):
            return [a, b]
    return [format_time(start_s, 3), format_time(end_s, 3)]


# --------------------------------------------------------------------------- session


def _digest(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


class LiveSession:
    """Owns live.toml. Runs on the Tk thread only, so it needs no locks.

    ``text`` is the last valid text (what the player is doing); ``disk_text`` is whatever was
    last seen on disk (which may be invalid). ``revision`` bumps whenever the disk changed
    behind our back.
    """

    def __init__(self, path: Path | str, song_map: SongMap) -> None:
        self.path = Path(path)
        self.song_map = song_map
        self.state = LiveState()
        self.text = ""
        self.disk_text = ""
        self.error: Optional[LiveParseError] = None
        self.message = ""
        self.revision = 0
        self._disk_hash: Optional[str] = None
        self._sig: Optional[tuple[int, int]] = None
        self._write_pending: Optional[str] = None

        if not self.path.exists():
            self._write(starter_text(song_map))
        raw = self.path.read_bytes()
        self._remember(raw)
        text = self._decode(raw)
        self.disk_text = text
        try:
            self.state = parse_live(text, song_map)
            self.text = text
        except LiveSyntaxError as exc:
            self.error = exc.error
            self.text = starter_text(song_map)

    # -- helpers

    @staticmethod
    def _decode(raw: bytes) -> str:
        return raw.decode("utf-8", errors="replace").replace("\r\n", "\n")

    def _remember(self, raw: bytes) -> None:
        self._disk_hash = _digest(raw)
        try:
            st = self.path.stat()
            self._sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._sig = None

    def _write(self, text: str) -> bool:
        """Atomic write (tmp + os.replace). On a locked file, retry on the next poll."""
        raw = text.encode("utf-8")
        if _digest(raw) == self._disk_hash and self.path.exists():
            self._write_pending = None
            return True
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_bytes(raw)
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02)
        except OSError as exc:
            self._write_pending = text
            self.message = f"could not write {self.path.name}: {exc}"
            return False
        self._write_pending = None
        self.message = ""
        self.disk_text = text
        self._remember(raw)
        return True

    # -- API

    def set_text(self, text: str) -> LiveState | LiveParseError:
        """Called by the panel. Valid text becomes current and is written to disk."""
        text = text.replace("\r\n", "\n")
        try:
            state = parse_live(text, self.song_map)
        except LiveSyntaxError as exc:
            return exc.error
        self.text = text
        self.state = state
        self.error = None
        self._write(text)
        return state

    def apply_ui(self, **changes: Any) -> Optional[str]:
        """Edit the last valid text with UI changes; returns the new text (or None on failure)."""
        try:
            new = edit_text(self.text, **changes)
        except LiveSyntaxError:
            return None
        if new == self.text:
            return new
        result = self.set_text(new)
        return None if isinstance(result, LiveParseError) else new

    def update(self, **changes: Any) -> Optional[str]:
        """Like apply_ui, but re-reads the file first and refuses to touch a file with errors."""
        self.poll()
        if self.error is not None:
            self.message = "live.toml has errors — UI change not saved"
            return None
        return self.apply_ui(**changes)

    def poll(self) -> Optional[LiveState]:
        """Check the file for outside edits. Returns a new state only if something changed."""
        if self._write_pending is not None:
            self._write(self._write_pending)
        try:
            st = self.path.stat()
        except OSError:
            return None  # mid-rename on an atomic save: retry next poll
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._sig:
            return None
        try:
            raw = self.path.read_bytes()
        except OSError:
            return None  # locked: retry next poll (sig not updated)
        if not raw.strip():
            return None  # truncated mid-save: wait for the content
        self._sig = sig
        digest = _digest(raw)
        if digest == self._disk_hash:
            return None  # our own write, or an identical re-save
        self._disk_hash = digest
        text = self._decode(raw)
        self.disk_text = text
        self.revision += 1
        try:
            state = parse_live(text, self.song_map)
        except LiveSyntaxError as exc:
            self.error = exc.error
            return None
        self.error = None
        self.text = text
        self.state = state
        return state
