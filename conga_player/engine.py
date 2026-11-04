"""Audio engine for conga_player using pedalboard AudioStream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Optional
import numpy as np

from pedalboard import Gain, Pedalboard
from pedalboard.io import AudioStream


@dataclass
class EngineState:
    position_s: float
    playing: bool
    loop_enabled: bool
    loop_start_s: float
    loop_end_s: float
    gain_db: float
    frames_written: int
    region_index: int = -1
    region_count: int = 0
    regions_s: tuple = field(default_factory=tuple)


class Engine:
    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        buffer_size: int = 2048,
        output_device_name: Optional[str] = None,
        declick_ms: float = 2.0,
        latency_offset_ms: Optional[float] = None,
    ) -> None:
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            # Transpose if passed (frames, channels)
            audio = audio.T

        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.num_channels = self.audio.shape[0]
        self.total_frames = self.audio.shape[1]
        self.sample_rate = int(sample_rate)
        self.buffer_size = int(buffer_size)
        self.output_device_name = output_device_name
        self.declick_ms = float(declick_ms)
        self.latency_offset_ms = latency_offset_ms

        self._lock = threading.Lock()
        self._pos_frame = 0
        self._playing = False
        # Active play regions (start_frame, end_frame), already expanded by repeat.
        # Empty list = free play. One region = a classic loop.
        self._regions: list[tuple[int, int]] = []
        self._region_idx = 0
        self._loop_bounds: Optional[tuple[int, int]] = None
        self._fade_in_next = False
        self._gain_db = 0.0
        self._frames_written = 0

        self._last_write_mono = time.monotonic()
        self._last_write_song_frame = 0
        self._last_write_playing = False
        # (mono_time, block_len, [(out_offset, song_frame, length), ...]) per played block
        self._history: deque = deque(maxlen=8)

        self._quit = False
        self._thread: Optional[threading.Thread] = None
        self._next_write = 0.0

        self.gain_plugin = Gain(gain_db=0.0)
        self.board = Pedalboard([self.gain_plugin])

    def start(self) -> None:
        """Start the background audio feeder thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._quit = False
        self._thread = threading.Thread(target=self._feed_loop, daemon=True, name="AudioFeeder")
        self._thread.start()

    def close(self) -> None:
        """Stop playback and terminate the feeder thread."""
        with self._lock:
            self._quit = True
            self._playing = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def play(self) -> None:
        """Resume playback."""
        with self._lock:
            if not self._playing:
                self._history.clear()
                self._playing = True
                self._last_write_mono = time.monotonic()
                self._last_write_song_frame = self._pos_frame
                self._last_write_playing = True

    def pause(self) -> None:
        """Pause playback."""
        with self._lock:
            self._playing = False
            self._last_write_playing = False

    def toggle(self) -> None:
        """Toggle playback between play and pause."""
        with self._lock:
            if self._playing:
                self._playing = False
                self._last_write_playing = False
            else:
                self._history.clear()
                self._playing = True
                self._last_write_mono = time.monotonic()
                self._last_write_song_frame = self._pos_frame
                self._last_write_playing = True

    def stop(self) -> None:
        """Stop playback and rewind to beginning."""
        with self._lock:
            self._playing = False
            self._last_write_playing = False
            self._pos_frame = 0
            self._last_write_song_frame = 0
            self._history.clear()

    def seek(self, seconds: float) -> None:
        """Seek to a position in seconds."""
        with self._lock:
            frame = max(0, min(int(round(seconds * self.sample_rate)), self.total_frames))
            self._pos_frame = frame
            self._last_write_mono = time.monotonic()
            self._last_write_song_frame = frame
            self._history.clear()
            self._fade_in_next = False

    def _to_frame(self, seconds: float) -> int:
        return max(0, min(int(round(seconds * self.sample_rate)), self.total_frames))

    def set_sequence(self, regions_s: list[tuple[float, float]]) -> None:
        """Play the given (start_s, end_s) regions one after another, wrapping at the end.

        Repeats are expressed by listing a region several times. An empty list means free play.
        If the playhead is inside one of the new regions it keeps playing without a gap,
        otherwise it jumps to the start of the first region (declicked).
        """
        regions: list[tuple[int, int]] = []
        for a, b in regions_s:
            fa, fb = self._to_frame(a), self._to_frame(b)
            if fa > fb:
                fa, fb = fb, fa
            if fb > fa:
                regions.append((fa, fb))
        with self._lock:
            if regions == self._regions:
                return
            self._regions = regions
            if not regions:
                self._region_idx = 0
                return
            for i, (s, e) in enumerate(regions):
                if s <= self._pos_frame < e:
                    self._region_idx = i
                    return
            self._region_idx = 0
            self._pos_frame = regions[0][0]
            self._history.clear()
            self._last_write_mono = time.monotonic()
            self._last_write_song_frame = self._pos_frame
            self._fade_in_next = True

    def set_loop(self, start_s: float, end_s: float) -> None:
        """Set loop boundaries in seconds and enable looping."""
        a, b = self._to_frame(start_s), self._to_frame(end_s)
        if a > b:
            a, b = b, a
        with self._lock:
            self._loop_bounds = (a, b)
        self.set_sequence([(a / float(self.sample_rate), b / float(self.sample_rate))])

    def clear_loop(self) -> None:
        """Clear loop boundaries and disable looping."""
        with self._lock:
            self._loop_bounds = None
        self.set_sequence([])

    def set_loop_enabled(self, enabled: bool) -> None:
        """Enable or disable looping without changing boundaries."""
        with self._lock:
            bounds = self._loop_bounds
        if enabled and bounds is not None:
            sr = float(self.sample_rate)
            self.set_sequence([(bounds[0] / sr, bounds[1] / sr)])
        else:
            self.set_sequence([])
    def set_gain_db(self, db: float) -> None:
        """Set output volume gain in decibels."""
        with self._lock:
            self._gain_db = float(db)
            self.gain_plugin.gain_db = self._gain_db

    def _locate(self, history: list, t: float) -> Optional[tuple[int, int]]:
        """Map monotonic time t to (song_frame, region_idx) using the recorded block history."""
        entry = None
        for item in reversed(history):
            if item[0] <= t:
                entry = item
                break
        if entry is None:
            return None
        mono, _blk_n, runs = entry
        if not runs:
            return None
        off = int(round((t - mono) * self.sample_rate))
        for o, f, ln, ridx in runs:
            if o <= off < o + ln:
                return f + off - o, ridx
        o, f, ln, ridx = runs[-1]
        return f + (off - o), ridx

    def state(self) -> EngineState:
        """Query engine state with smooth, seam-accurate position interpolation."""
        with self._lock:
            playing = self._playing
            pos_frame = self._pos_frame
            regions = list(self._regions)
            region_idx = self._region_idx
            gain_db = self._gain_db
            frames_written = self._frames_written
            last_mono = self._last_write_mono
            last_song_frame = self._last_write_song_frame
            last_playing = self._last_write_playing
            history = list(self._history)

        sr = float(self.sample_rate)
        latency_frames = (
            int(round(self.latency_offset_ms * self.sample_rate / 1000.0))
            if self.latency_offset_ms is not None
            else self.buffer_size
        )

        cur_region = region_idx
        if playing and last_playing:
            now = time.monotonic()
            found = self._locate(history, now - latency_frames / sr)
            if found is not None:
                cur_frame, cur_region = found
            else:
                elapsed = now - last_mono
                cur_frame = last_song_frame + int(round(elapsed * self.sample_rate)) - latency_frames
            cur_frame = max(0, min(cur_frame, self.total_frames))
            pos_s = cur_frame / sr
        else:
            pos_s = pos_frame / sr

        if regions:
            cur_region = max(0, min(cur_region, len(regions) - 1))
            loop_start_s = regions[0][0] / sr
            loop_end_s = regions[0][1] / sr
        else:
            cur_region = -1
            loop_start_s = 0.0
            loop_end_s = self.total_frames / sr

        return EngineState(
            position_s=pos_s,
            playing=playing,
            loop_enabled=bool(regions),
            loop_start_s=loop_start_s,
            loop_end_s=loop_end_s,
            gain_db=gain_db,
            frames_written=frames_written,
            region_index=cur_region,
            region_count=len(regions),
            regions_s=tuple((s / sr, e / sr) for s, e in regions),
        )

    def _next_block(self, n: int) -> np.ndarray:
        """Assemble the next block of audio frames.

        Pure and lock-protected: safe to drive directly in tests without an AudioStream.
        """
        with self._lock:
            song_frame_at_start = self._pos_frame
            was_playing = self._playing

            if not self._playing:
                self._last_write_song_frame = song_frame_at_start
                self._last_write_mono = time.monotonic()
                self._last_write_playing = False
                self._frames_written += n
                return np.zeros((self.num_channels, n), dtype=np.float32)

            out = np.zeros((self.num_channels, n), dtype=np.float32)
            written = 0
            declick_frames = int(round(self.declick_ms * self.sample_rate / 1000.0))
            wrap_seams: list[int] = []
            runs: list[tuple[int, int, int, int]] = []
            regions = self._regions

            if self._fade_in_next:
                wrap_seams.append(0)
                self._fade_in_next = False

            while written < n and self._playing:
                if regions:
                    if self._region_idx >= len(regions):
                        self._region_idx = 0
                    start, end = regions[self._region_idx]
                    if not (start <= self._pos_frame < end):
                        found = None
                        for i, (rs, re_) in enumerate(regions):
                            if rs <= self._pos_frame < re_:
                                found = i
                                break
                        if found is None:
                            found = 0
                            self._pos_frame = regions[0][0]
                        self._region_idx = found
                        start, end = regions[found]
                        if written > 0:
                            wrap_seams.append(written)

                    take = min(n - written, end - self._pos_frame)
                    out[:, written : written + take] = self.audio[:, self._pos_frame : self._pos_frame + take]
                    runs.append((written, self._pos_frame, take, self._region_idx))
                    self._pos_frame += take
                    written += take

                    if self._pos_frame >= end:
                        self._region_idx = (self._region_idx + 1) % len(regions)
                        self._pos_frame = regions[self._region_idx][0]
                        wrap_seams.append(written)
                        if written >= n:
                            # Seam falls exactly on the block edge: fade the next block in.
                            self._fade_in_next = True
                else:
                    available = self.total_frames - self._pos_frame
                    if available <= 0:
                        self._playing = False
                        break
                    take = min(n - written, available)
                    out[:, written : written + take] = self.audio[:, self._pos_frame : self._pos_frame + take]
                    runs.append((written, self._pos_frame, take, -1))
                    self._pos_frame += take
                    written += take
                    if self._pos_frame >= self.total_frames:
                        self._playing = False
                        break

            # Apply declick fades at seam locations
            if declick_frames > 0 and wrap_seams:
                prev_seam = 0
                for idx, seam in enumerate(wrap_seams):
                    next_seam = wrap_seams[idx + 1] if idx + 1 < len(wrap_seams) else n

                    fade_out_len = min(declick_frames, seam - prev_seam)
                    if fade_out_len > 0:
                        ramp_out = np.linspace(1.0, 0.0, fade_out_len + 1, dtype=np.float32)[1:]
                        out[:, seam - fade_out_len : seam] *= ramp_out

                    fade_in_len = min(declick_frames, next_seam - seam)
                    if fade_in_len > 0:
                        ramp_in = np.linspace(0.0, 1.0, fade_in_len + 1, dtype=np.float32)[:-1]
                        out[:, seam : seam + fade_in_len] *= ramp_in

                    prev_seam = seam

            # Apply gain if non-zero
            if self._gain_db != 0.0:
                self.gain_plugin.gain_db = self._gain_db
                out = self.board(out, self.sample_rate, reset=False)

            now = time.monotonic()
            self._history.append((now, n, runs))
            self._last_write_song_frame = song_frame_at_start
            self._last_write_mono = now
            self._last_write_playing = was_playing
            self._frames_written += n

            return out

    def _feed_loop(self) -> None:
        """Audio feeder loop running in dedicated thread."""
        dev = self.output_device_name
        if dev is None:
            dev = AudioStream.default_output_device_name
        elif isinstance(dev, int) or (isinstance(dev, str) and dev.strip().isdigit()):
            idx = int(dev)
            devs = AudioStream.output_device_names
            if 0 <= idx < len(devs):
                dev = devs[idx]

        try:
            stream_ctx = AudioStream(
                output_device_name=dev,
                num_input_channels=0,
                num_output_channels=self.num_channels,
                sample_rate=self.sample_rate,
                buffer_size=self.buffer_size,
            )
        except Exception:
            # Fallback if specific sample rate is rejected
            stream_ctx = AudioStream(
                output_device_name=dev,
                num_input_channels=0,
                num_output_channels=self.num_channels,
                sample_rate=None,
                buffer_size=self.buffer_size,
            )

        with stream_ctx as stream:
            self.sample_rate = int(stream.sample_rate)
            self._next_write = time.monotonic()

            while not self._quit:
                now = time.monotonic()
                if self._next_write - now > 0.002:
                    time.sleep(self._next_write - now - 0.002)

                block = self._next_block(self.buffer_size)

                try:
                    stream.write(block, self.sample_rate)
                except Exception:
                    break

                now = time.monotonic()
                self._next_write = max(self._next_write + self.buffer_size / float(self.sample_rate), now)
