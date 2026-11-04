"""Tkinter user interface for conga_player."""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np

from conga_player.engine import Engine
from conga_player.live import (
    LiveSession,
    LiveState,
    expand_regions,
    loop_spec_for_bars,
    loop_spec_for_phrase,
    loop_spec_for_range,
    region_to_segment_map,
)
from conga_player.peaks import zoom_peaks
from conga_player.songmap import Phrase, SongMap
from conga_player.toml_panel import TomlPanel

CANVAS_WIDTH = 1200

# Kind -> Hex color mapping for phrases
KIND_COLORS = {
    1: "#3b82f6",  # Intro (Blue)
    2: "#10b981",  # Verse 1 (Emerald)
    3: "#059669",  # Verse 2 (Dark Emerald)
    4: "#14b8a6",  # Verse 3 (Teal)
    5: "#06b6d4",  # Verse 4 (Cyan)
    6: "#6366f1",  # Verse 5 (Indigo)
    7: "#8b5cf6",  # Verse 6 (Violet)
    8: "#f59e0b",  # Bridge (Amber)
    9: "#ef4444",  # Chorus (Red)
    10: "#64748b",  # Outro (Slate)
}
DEFAULT_PHRASE_COLOR = "#a855f7"

ZOOM_LEVELS = [4, 8, 16, 32]
SNAP_MODES = ["beat", "bar", "phrase", "off"]


class CongaPlayerApp:
    def __init__(
        self, song_map: SongMap, engine: Engine, audio: np.ndarray, live: Optional[LiveSession] = None
    ) -> None:
        self.song_map = song_map
        self.engine = engine
        self.audio = audio
        self.sample_rate = engine.sample_rate
        self.live = live
        self.panel: Optional[TomlPanel] = None

        # Live session state
        self._live_state: Optional[LiveState] = None
        self._mode = "free"
        self._loop_seg: Optional[tuple[float, float]] = None
        self._region_to_seg: list[int] = []
        self._tick_n = 0
        self._last_playing = False
        self._vol_after: Optional[str] = None
        self._suppress_vol = False
        self._overlay_key: Optional[tuple] = None
        self._last_status_err: Optional[str] = None

        self.zoom_bars = 8
        self.snap_mode = "beat"

        # Zoom window caching
        self._cached_zoom_bar = -999
        self._cached_zoom_bars_len = self.zoom_bars
        self._zoom_win_start = 0.0
        self._zoom_win_end = 1.0

        # Drag state for setting loops
        self._drag_start_t: Optional[float] = None
        self._dragging_canvas: Optional[str] = None

        # Build window
        self.root = tk.Tk()
        self.root.title(f"conga_player — {song_map.artist} - {song_map.title}")
        if live is not None:
            self.root.geometry("1240x820")
            self.root.resizable(True, True)
        else:
            self.root.geometry("1240x480")
            self.root.resizable(False, False)
        self.root.configure(bg="#0f172a")

        self._init_styles()
        self._build_ui()
        self._bind_keys()

    def _init_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#0f172a", foreground="#f8fafc")
        style.configure("TLabel", background="#0f172a", foreground="#f8fafc", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#38bdf8")
        style.configure("Status.TLabel", font=("Consolas", 11), foreground="#38bdf8", background="#090d16")
        style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            background="#1e293b",
            foreground="#f8fafc",
            borderwidth=1,
            focuscolor="none",
        )
        style.map(
            "TButton",
            background=[("active", "#334155"), ("pressed", "#475569")],
            foreground=[("active", "#38bdf8")],
        )

    def _build_ui(self) -> None:
        paned: Optional[ttk.PanedWindow] = None
        if self.live is not None:
            paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
            paned.pack(fill=tk.BOTH, expand=True)
            self.top = tk.Frame(paned, bg="#0f172a")
            paned.add(self.top, weight=0)
        else:
            self.top = self.root

        # 1. Header Frame
        header_frame = tk.Frame(self.top, bg="#0f172a", pady=6)
        header_frame.pack(fill=tk.X, padx=20)

        header_text = (
            f"{self.song_map.artist} — {self.song_map.title}    "
            f"Tempo: {self.song_map.bpm:.1f} BPM    Key: {self.song_map.key or 'N/A'}    "
            f"Grid: {self.song_map.grid_source}"
        )
        header_lbl = ttk.Label(header_frame, text=header_text, style="Header.TLabel")
        header_lbl.pack(side=tk.LEFT)

        # 2. Main Canvases Container (centered with 20px margins -> 1200px wide)
        canvas_container = tk.Frame(self.top, bg="#0f172a")
        canvas_container.pack(fill=tk.X, padx=20)

        # A. Phrase Lane (h=28)
        self.phrase_canvas = tk.Canvas(
            canvas_container,
            width=CANVAS_WIDTH,
            height=28,
            bg="#090d16",
            highlightthickness=1,
            highlightbackground="#1e293b",
        )
        self.phrase_canvas.pack(pady=(0, 2))
        self._draw_phrase_lane()

        # C. Bars Ruler Lane (h=18)
        self.bars_canvas = tk.Canvas(
            canvas_container,
            width=CANVAS_WIDTH,
            height=18,
            bg="#0f172a",
            highlightthickness=0,
        )
        self.bars_canvas.pack(pady=(1, 4))
        self._draw_bars_lane()

        # D. Zoom Lane (h=110)
        self.zoom_canvas = tk.Canvas(
            canvas_container,
            width=CANVAS_WIDTH,
            height=110,
            bg="#090d16",
            highlightthickness=1,
            highlightbackground="#1e293b",
        )
        self.zoom_canvas.pack(pady=(2, 6))

        # 3. Transport Controls Row
        ctrl_frame = tk.Frame(self.top, bg="#0f172a")
        ctrl_frame.pack(fill=tk.X, padx=20, pady=4)

        self.btn_play = ttk.Button(ctrl_frame, text="▶ Play", width=8, command=self._on_toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=3)

        self.btn_stop = ttk.Button(ctrl_frame, text="■ Stop", width=7, command=self._on_stop)
        self.btn_stop.pack(side=tk.LEFT, padx=3)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.btn_loop = ttk.Button(ctrl_frame, text="Loop: OFF", width=10, command=self._on_toggle_loop)
        self.btn_loop.pack(side=tk.LEFT, padx=3)

        self.btn_seq = ttk.Button(ctrl_frame, text="Seq", width=5, command=self._on_sequence)
        self.btn_seq.pack(side=tk.LEFT, padx=3)

        for bars in (1, 2, 4, 8, 16):
            btn = ttk.Button(
                ctrl_frame,
                text=f"{bars}b",
                width=4,
                command=lambda b=bars: self._on_loop_bars(b),
            )
            btn.pack(side=tk.LEFT, padx=2)

        self.btn_phrase_loop = ttk.Button(ctrl_frame, text="⟲ Phrase", width=9, command=self._on_loop_phrase)
        self.btn_phrase_loop.pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.btn_snap = ttk.Button(
            ctrl_frame,
            text=f"Snap: {self.snap_mode.capitalize()}",
            width=12,
            command=self._on_cycle_snap,
        )
        self.btn_snap.pack(side=tk.LEFT, padx=3)

        ttk.Label(ctrl_frame, text="Zoom:").pack(side=tk.LEFT, padx=(6, 2))
        self.btn_zoom_out = ttk.Button(ctrl_frame, text="-", width=3, command=self._on_zoom_out)
        self.btn_zoom_out.pack(side=tk.LEFT, padx=1)
        self.lbl_zoom = ttk.Label(ctrl_frame, text=f"{self.zoom_bars}b", width=4)
        self.lbl_zoom.pack(side=tk.LEFT, padx=1)
        self.btn_zoom_in = ttk.Button(ctrl_frame, text="+", width=3, command=self._on_zoom_in)
        self.btn_zoom_in.pack(side=tk.LEFT, padx=1)

        # Volume Slider
        vol_frame = tk.Frame(ctrl_frame, bg="#0f172a")
        vol_frame.pack(side=tk.RIGHT, padx=4)
        ttk.Label(vol_frame, text="Vol:").pack(side=tk.LEFT, padx=(0, 4))
        self.vol_slider = ttk.Scale(
            vol_frame,
            from_=-24.0,
            to=6.0,
            value=0.0,
            orient=tk.HORIZONTAL,
            length=100,
            command=self._on_volume_change,
        )
        self.vol_slider.pack(side=tk.LEFT)
        self.lbl_vol = ttk.Label(vol_frame, text="0 dB", width=6)
        self.lbl_vol.pack(side=tk.LEFT, padx=(4, 0))

        # 4. Status Bar Row
        status_frame = tk.Frame(self.top, bg="#090d16", relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, padx=20, pady=(4, 8))

        self.live_lbl = ttk.Label(status_frame, text="", style="Status.TLabel", padding=(8, 4))
        self.live_lbl.pack(side=tk.RIGHT)

        self.status_lbl = ttk.Label(
            status_frame,
            text="00:00.000   bar 1.1   beat 1   phrase --   loop off",
            style="Status.TLabel",
            padding=(8, 4),
        )
        self.status_lbl.pack(fill=tk.X)

        # Dynamic canvas items
        self.phrase_playhead = self.phrase_canvas.create_line(
            0, 0, 0, 28, fill="#ffffff", width=2
        )

        self.zoom_playhead = self.zoom_canvas.create_line(
            0, 0, 0, 110, fill="#ffffff", width=2
        )

        self._bind_mouse_events()

        if paned is not None and self.live is not None:
            self.panel = TomlPanel(paned, self.live, self._apply_live, on_escape=self._focus_lanes)
            paned.add(self.panel, weight=1)

    def _draw_phrase_lane(self) -> None:
        self.phrase_canvas.delete("all")
        dur = self.song_map.duration_s
        if dur <= 0:
            return

        for p in self.song_map.phrases:
            x1 = (p.start_s / dur) * CANVAS_WIDTH
            x2 = (p.end_s / dur) * CANVAS_WIDTH
            color = KIND_COLORS.get(p.kind, DEFAULT_PHRASE_COLOR)

            tag = f"phrase_{p.index}"
            self.phrase_canvas.create_rectangle(
                x1, 0, x2, 28, fill=color, outline="#0f172a", width=1, tags=(tag, "phrase_rect")
            )
            width = x2 - x1
            if width >= 26:
                label_txt = p.label
                if width < 45 and len(label_txt) > 4:
                    label_txt = label_txt[:3]
                self.phrase_canvas.create_text(
                    (x1 + x2) / 2,
                    14,
                    text=label_txt,
                    fill="#ffffff",
                    font=("Segoe UI", 8, "bold"),
                    tags=(tag, "phrase_text"),
                )

    def _draw_bars_lane(self) -> None:
        self.bars_canvas.delete("all")
        dur = self.song_map.duration_s
        if dur <= 0:
            return

        for i, bar_t in enumerate(self.song_map.bar_times_s):
            x = (bar_t / dur) * CANVAS_WIDTH
            bar_num = i + 1
            if (bar_num - 1) % 16 == 0:
                self.bars_canvas.create_line(x, 0, x, 14, fill="#94a3b8", width=1)
                self.bars_canvas.create_text(
                    x + 2, 7, text=str(bar_num), fill="#94a3b8", font=("Consolas", 8), anchor="w"
                )
            else:
                self.bars_canvas.create_line(x, 0, x, 5, fill="#475569", width=1)

    def _update_zoom_lane(self, cur_pos_s: float, force_redraw: bool = False) -> None:
        cur_bar = self.song_map.bar_index_at(cur_pos_s)
        if not force_redraw and cur_bar == self._cached_zoom_bar and self.zoom_bars == self._cached_zoom_bars_len:
            return

        self._cached_zoom_bar = cur_bar
        self._cached_zoom_bars_len = self.zoom_bars

        # Determine bar window: centered on cur_bar
        half = self.zoom_bars // 2
        total_bars = len(self.song_map.bar_times_s)
        start_bar_idx = max(0, cur_bar - half)
        end_bar_idx = min(total_bars - 1, start_bar_idx + self.zoom_bars)
        if end_bar_idx - start_bar_idx < self.zoom_bars:
            start_bar_idx = max(0, end_bar_idx - self.zoom_bars)

        t_start = float(self.song_map.bar_times_s[start_bar_idx])
        if end_bar_idx + 1 < total_bars:
            t_end = float(self.song_map.bar_times_s[end_bar_idx + 1])
        else:
            t_end = float(self.song_map.duration_s)

        if t_end <= t_start:
            t_end = t_start + 1.0

        self._zoom_win_start = t_start
        self._zoom_win_end = t_end

        # Redraw zoom background items
        self.zoom_canvas.delete("zoom_bg")

        # 1. Peaks
        peaks = zoom_peaks(self.audio, self.sample_rate, t_start, t_end, CANVAS_WIDTH)
        cy = 55.0
        max_h = 48.0
        for x in range(CANVAS_WIDTH):
            h = peaks[x] * max_h
            if h > 0.5:
                self.zoom_canvas.create_line(
                    x, cy - h, x, cy + h, fill="#38bdf8", width=1, tags="zoom_bg"
                )

        # 2. Bar and beat gridlines
        # Find all beats in [t_start, t_end]
        b_first = int(np.searchsorted(self.song_map.beat_times_s, t_start))
        b_last = int(np.searchsorted(self.song_map.beat_times_s, t_end))
        win_dur = t_end - t_start

        for b_idx in range(b_first, min(b_last + 1, len(self.song_map.beat_times_s))):
            bt = self.song_map.beat_times_s[b_idx]
            if bt < t_start or bt > t_end:
                continue
            bx = ((bt - t_start) / win_dur) * CANVAS_WIDTH
            b_num = self.song_map.beat_numbers[b_idx]

            if b_num == 1:
                # Downbeat / Bar line
                bar_idx = self.song_map.bar_index_at(bt)
                self.zoom_canvas.create_line(bx, 0, bx, 110, fill="#ef4444", width=1, tags="zoom_bg")
                self.zoom_canvas.create_text(
                    bx + 4,
                    12,
                    text=f"| {bar_idx + 1}",
                    fill="#fca5a5",
                    font=("Consolas", 9, "bold"),
                    anchor="w",
                    tags="zoom_bg",
                )
            else:
                # Secondary beat ticks
                self.zoom_canvas.create_line(bx, 45, bx, 65, fill="#334155", width=1, tags="zoom_bg")
                self.zoom_canvas.create_text(
                    bx, 12, text="·", fill="#64748b", font=("Consolas", 10), tags="zoom_bg"
                )

        # Ensure segment overlays and playhead are on top of background
        self.zoom_canvas.tag_raise("seg")
        self.zoom_canvas.tag_raise(self.zoom_playhead)

    def _bind_mouse_events(self) -> None:
        # Phrase Canvas
        self.phrase_canvas.bind("<Button-1>", self._on_phrase_click)
        self.phrase_canvas.bind("<Double-Button-1>", self._on_phrase_double_click)

        # Zoom Canvas
        self.zoom_canvas.bind("<ButtonPress-1>", lambda e: self._on_drag_press(e, "zoom"))
        self.zoom_canvas.bind("<B1-Motion>", lambda e: self._on_drag_motion(e, "zoom"))
        self.zoom_canvas.bind("<ButtonRelease-1>", lambda e: self._on_drag_release(e, "zoom"))

    def _panel_has_focus(self) -> bool:
        return self.panel is not None and self.root.focus_get() is self.panel.text

    def _bind_keys(self) -> None:
        def bind(seq: str, fn) -> None:
            # While the TOML panel has focus, keystrokes go into the text only.
            self.root.bind(seq, lambda e: None if self._panel_has_focus() else fn())

        bind("<space>", self._on_toggle_play)
        bind("<KeyPress-s>", self._on_stop)
        bind("<KeyPress-S>", self._on_stop)
        bind("<KeyPress-l>", self._on_toggle_loop)
        bind("<KeyPress-L>", self._on_toggle_loop)
        bind("<KeyPress-q>", self._on_sequence)
        bind("<KeyPress-Q>", self._on_sequence)
        bind("<Escape>", self._on_clear_loop)
        bind("<KeyPress-p>", self._on_loop_phrase)
        bind("<KeyPress-P>", self._on_loop_phrase)

        bind("1", lambda: self._on_loop_bars(1))
        bind("2", lambda: self._on_loop_bars(2))
        bind("4", lambda: self._on_loop_bars(4))
        bind("8", lambda: self._on_loop_bars(8))

        bind("<bracketleft>", self._on_prev_phrase)
        bind("<bracketright>", self._on_next_phrase)

        bind("<Left>", lambda: self._on_seek_step(-1, beat=False))
        bind("<Right>", lambda: self._on_seek_step(1, beat=False))
        bind("<Shift-Left>", lambda: self._on_seek_step(-1, beat=True))
        bind("<Shift-Right>", lambda: self._on_seek_step(1, beat=True))

        bind("<minus>", self._on_zoom_out)
        bind("<underscore>", self._on_zoom_out)
        bind("<plus>", self._on_zoom_in)
        bind("<equal>", self._on_zoom_in)

    def _focus_lanes(self) -> None:
        self.zoom_canvas.focus_set()

    # --- Mouse Handlers ---

    def _x_to_time(self, x: float, canvas_name: str) -> float:
        x_clamped = max(0.0, min(float(x), float(CANVAS_WIDTH)))
        if canvas_name == "zoom":
            dur = self._zoom_win_end - self._zoom_win_start
            return self._zoom_win_start + (x_clamped / CANVAS_WIDTH) * dur
        return 0.0

    def _on_phrase_click(self, event: tk.Event) -> None:
        self._focus_lanes()
        dur = self.song_map.duration_s
        t = (max(0, min(event.x, CANVAS_WIDTH)) / CANVAS_WIDTH) * dur
        p = self.song_map.phrase_at(t)
        if p:
            self.engine.seek(p.start_s)

    def _on_phrase_double_click(self, event: tk.Event) -> None:
        dur = self.song_map.duration_s
        t = (max(0, min(event.x, CANVAS_WIDTH)) / CANVAS_WIDTH) * dur
        p = self.song_map.phrase_at(t)
        if p:
            self._set_loop(p.start_s, p.end_s, loop_spec_for_phrase(self.song_map, p))

    def _on_drag_press(self, event: tk.Event, canvas_name: str) -> None:
        self._focus_lanes()
        t = self._x_to_time(event.x, canvas_name)
        snapped_t = self.song_map.snap(t, self.snap_mode)
        self._drag_start_t = snapped_t
        self._dragging_canvas = canvas_name

    def _on_drag_motion(self, event: tk.Event, canvas_name: str) -> None:
        if self._drag_start_t is None:
            return
        t = self._x_to_time(event.x, canvas_name)
        snapped_t = self.song_map.snap(t, self.snap_mode)
        if abs(snapped_t - self._drag_start_t) > 0.05:
            # Dynamically display loop region while dragging (written to live.toml on release)
            lo, hi = sorted((self._drag_start_t, snapped_t))
            self.engine.set_sequence([(lo, hi)])

    def _on_drag_release(self, event: tk.Event, canvas_name: str) -> None:
        if self._drag_start_t is None:
            return
        t = self._x_to_time(event.x, canvas_name)
        snapped_t = self.song_map.snap(t, self.snap_mode)
        if abs(snapped_t - self._drag_start_t) < 0.05:
            # Single click -> seek (seeks are not written to the file)
            self.engine.seek(snapped_t)
        else:
            lo, hi = sorted((self._drag_start_t, snapped_t))
            self._set_loop(lo, hi, loop_spec_for_range(self.song_map, lo, hi, self.snap_mode))

        self._drag_start_t = None
        self._dragging_canvas = None

    # --- Live session write-back ---

    def _live_update(self, **changes) -> None:
        """Write a UI action into live.toml (via the panel buffer when there is one)."""
        if self.live is None:
            return
        if self.panel is not None:
            state = self.panel.apply_ui(**changes)
        else:
            state = self.live.state if self.live.update(**changes) is not None else None
        if state is not None:
            self._apply_live(state)

    def _set_loop(self, start_s: float, end_s: float, spec) -> None:
        self._loop_seg = (start_s, end_s)
        self._mode = "loop"
        self.engine.set_sequence([(start_s, end_s)])
        self._live_update(mode="loop", loop=spec)

    def _set_playing(self, playing: bool) -> None:
        self._last_playing = playing
        self._live_update(playing=playing)

    def _set_slider(self, db: float) -> None:
        self._suppress_vol = True
        try:
            self.vol_slider.set(db)
        finally:
            self._suppress_vol = False
        self.lbl_vol.config(text=f"{db:+.1f} dB")

    def _apply_live(self, state: LiveState) -> None:
        """Make the engine and controls match a LiveState (from the file or the panel)."""
        self._live_state = state
        self._mode = state.mode
        self._loop_seg = (state.loop.start_s, state.loop.end_s) if state.loop else None

        self._set_slider(state.volume_db)
        self.engine.set_gain_db(state.volume_db)

        self._region_to_seg = []
        if state.mode == "sequence" and state.segments:
            self.engine.set_sequence(expand_regions(state.segments))
            self._region_to_seg = region_to_segment_map(state.segments)
        elif state.mode == "loop" and state.loop:
            self.engine.set_sequence([(state.loop.start_s, state.loop.end_s)])
        else:
            self.engine.set_sequence([])

        currently = self.engine.state().playing
        if state.playing and not currently:
            self.engine.play()
        elif not state.playing and currently:
            self.engine.pause()
        self._last_playing = state.playing

    # --- Button & Shortcut Handlers ---

    def _on_toggle_play(self) -> None:
        self.engine.toggle()
        self._set_playing(self.engine.state().playing)

    def _on_stop(self) -> None:
        self.engine.stop()
        self._set_playing(False)

    def _on_toggle_loop(self) -> None:
        if self._mode == "loop":
            self._mode = "free"
            self.engine.set_sequence([])
            self._live_update(mode="free")
        elif self._loop_seg is not None:
            self._mode = "loop"
            self.engine.set_sequence([self._loop_seg])
            self._live_update(mode="loop")

    def _on_sequence(self) -> None:
        if self.live is None:
            return
        self._live_update(mode="sequence")

    def _on_clear_loop(self) -> None:
        self._mode = "free"
        self._loop_seg = None
        self.engine.set_sequence([])
        self._live_update(mode="free", loop=None)

    def _on_loop_bars(self, n_bars: int) -> None:
        st = self.engine.state()
        cur_bar = self.song_map.bar_index_at(st.position_s)
        total_bars = len(self.song_map.bar_times_s)
        last_bar = min(total_bars, cur_bar + n_bars)  # 1-based, inclusive
        start_s = float(self.song_map.bar_times_s[cur_bar])
        if last_bar < total_bars:
            end_s = float(self.song_map.bar_times_s[last_bar])
        else:
            end_s = float(self.song_map.duration_s)
        self._set_loop(start_s, end_s, loop_spec_for_bars(cur_bar + 1, last_bar))

    def _on_loop_phrase(self) -> None:
        st = self.engine.state()
        p = self.song_map.phrase_at(st.position_s)
        if p:
            self._set_loop(p.start_s, p.end_s, loop_spec_for_phrase(self.song_map, p))

    def _on_cycle_snap(self) -> None:
        cur_idx = SNAP_MODES.index(self.snap_mode)
        self.snap_mode = SNAP_MODES[(cur_idx + 1) % len(SNAP_MODES)]
        self.btn_snap.config(text=f"Snap: {self.snap_mode.capitalize()}")

    def _on_zoom_in(self) -> None:
        idx = ZOOM_LEVELS.index(self.zoom_bars)
        if idx > 0:
            self.zoom_bars = ZOOM_LEVELS[idx - 1]
            self.lbl_zoom.config(text=f"{self.zoom_bars}b")
            self._update_zoom_lane(self.engine.state().position_s, force_redraw=True)

    def _on_zoom_out(self) -> None:
        idx = ZOOM_LEVELS.index(self.zoom_bars)
        if idx < len(ZOOM_LEVELS) - 1:
            self.zoom_bars = ZOOM_LEVELS[idx + 1]
            self.lbl_zoom.config(text=f"{self.zoom_bars}b")
            self._update_zoom_lane(self.engine.state().position_s, force_redraw=True)

    def _on_volume_change(self, val: str) -> None:
        db = float(val)
        self.lbl_vol.config(text=f"{db:+.1f} dB")
        self.engine.set_gain_db(db)
        if self._suppress_vol or self.live is None:
            return
        if self._vol_after is not None:
            self.root.after_cancel(self._vol_after)
        self._vol_after = self.root.after(300, lambda: self._write_volume(db))

    def _write_volume(self, db: float) -> None:
        self._vol_after = None
        self._live_update(volume_db=round(db, 1))

    def _on_prev_phrase(self) -> None:
        st = self.engine.state()
        p = self.song_map.phrase_at(st.position_s)
        if p:
            if st.position_s > p.start_s + 0.5:
                self.engine.seek(p.start_s)
            elif p.index > 1:
                prev_p = self.song_map.phrases[p.index - 2]
                self.engine.seek(prev_p.start_s)
        elif self.song_map.phrases:
            self.engine.seek(self.song_map.phrases[0].start_s)

    def _on_next_phrase(self) -> None:
        st = self.engine.state()
        p = self.song_map.phrase_at(st.position_s)
        if p and p.index < len(self.song_map.phrases):
            next_p = self.song_map.phrases[p.index]
            self.engine.seek(next_p.start_s)

    def _on_seek_step(self, step: int, beat: bool) -> None:
        st = self.engine.state()
        if beat:
            cur = self.song_map.beat_index_at(st.position_s)
            tgt = max(0, min(cur + step, len(self.song_map.beat_times_s) - 1))
            self.engine.seek(self.song_map.beat_times_s[tgt])
        else:
            cur = self.song_map.bar_index_at(st.position_s)
            tgt = max(0, min(cur + step, len(self.song_map.bar_times_s) - 1))
            self.engine.seek(self.song_map.bar_times_s[tgt])

    # --- Live polling, overlays & status ---

    def _poll_live(self) -> None:
        assert self.live is not None
        new_state = self.live.poll()
        if self.panel is not None:
            self.panel.check_external()
        if new_state is not None:
            self._apply_live(new_state)

        err: Optional[str] = None
        if self.panel is not None and self.panel.error:
            err = self.panel.error
        elif self.live.error is not None:
            err = str(self.live.error)
        elif self.live.state.errors:
            err = "\n".join(self.live.state.errors)
        elif self.live.message:
            err = self.live.message
        if err:
            self.live_lbl.config(text=f"live.toml \u2717 {err.splitlines()[0]}", foreground="#f87171")
        else:
            self.live_lbl.config(text="live.toml \u2713", foreground="#4ade80")
        if err != self._last_status_err:
            self._last_status_err = err
            if err:
                print(f"live.toml: {err}", file=sys.stderr)

    def _active_segment(self, region_index: int) -> Optional[int]:
        if 0 <= region_index < len(self._region_to_seg):
            return self._region_to_seg[region_index]
        return None

    def _draw_overlays(self, regions: tuple, region_index: int) -> None:
        dur = self.song_map.duration_s
        self.phrase_canvas.delete("seg")
        self.zoom_canvas.delete("seg")
        if dur <= 0 or not regions:
            return

        sequence = self._mode == "sequence" and len(self._region_to_seg) == len(regions)
        items: list[tuple[float, float, Optional[int], bool]] = []  # start, end, number, active
        if sequence:
            active = self._active_segment(region_index)
            seen: set[int] = set()
            for i, (s, e) in enumerate(regions):
                seg_i = self._region_to_seg[i]
                if seg_i in seen:
                    continue
                seen.add(seg_i)
                items.append((s, e, seg_i + 1, seg_i == active))
        else:
            items = [(s, e, None, True) for s, e in regions]

        win_dur = self._zoom_win_end - self._zoom_win_start
        for s, e, number, active in items:
            fill = "#7dd3fc" if active else "#38bdf8"
            stipple = "gray50" if active else "gray25"
            x1 = (s / dur) * CANVAS_WIDTH
            x2 = (e / dur) * CANVAS_WIDTH
            self.phrase_canvas.create_rectangle(
                x1, 0, x2, 28, fill=fill, stipple=stipple, outline="#38bdf8", tags="seg"
            )
            if number is not None:
                self.phrase_canvas.create_text(
                    x1 + 3, 14, text=str(number), anchor="w", fill="#ffffff", font=("Segoe UI", 8, "bold"), tags="seg"
                )
            if win_dur > 0:
                zx1 = max(0.0, ((s - self._zoom_win_start) / win_dur) * CANVAS_WIDTH)
                zx2 = min(float(CANVAS_WIDTH), ((e - self._zoom_win_start) / win_dur) * CANVAS_WIDTH)
                if zx2 > zx1:
                    self.zoom_canvas.create_rectangle(
                        zx1, 0, zx2, 110, fill=fill, stipple=stipple, outline="#38bdf8", tags="seg"
                    )
                    if number is not None:
                        self.zoom_canvas.create_text(
                            zx1 + 3, 100, text=str(number), anchor="w", fill="#ffffff",
                            font=("Segoe UI", 9, "bold"), tags="seg",
                        )
        self.phrase_canvas.tag_raise(self.phrase_playhead)
        self.zoom_canvas.tag_raise(self.zoom_playhead)

    # --- Tick Loop ---

    def _tick(self) -> None:
        st = self.engine.state()
        dur = self.song_map.duration_s
        self._tick_n += 1

        if self.live is not None:
            if self._tick_n % 5 == 0:
                self._poll_live()
            if self._last_playing and not st.playing:
                # The engine stopped on its own (end of track).
                self._set_playing(False)

        loop_on = self._mode == "loop" and st.loop_enabled

        # Update Play/Pause button text
        self.btn_play.config(text="❚❚ Pause" if st.playing else "▶ Play")
        self.btn_loop.config(text="Loop: ON" if loop_on else "Loop: OFF")

        # 1. Update phrase-lane playhead
        if dur > 0:
            px = (st.position_s / dur) * CANVAS_WIDTH
            self.phrase_canvas.coords(self.phrase_playhead, px, 0, px, 28)

        # 2. Update Zoom lane background & playhead
        self._update_zoom_lane(st.position_s)
        win_dur = self._zoom_win_end - self._zoom_win_start
        if win_dur > 0:
            zpx = ((st.position_s - self._zoom_win_start) / win_dur) * CANVAS_WIDTH
            self.zoom_canvas.coords(self.zoom_playhead, zpx, 0, zpx, 110)

        # 3. Segment overlays: redraw only when something changed
        sequence = self._mode == "sequence" and len(self._region_to_seg) == st.region_count > 0
        key = (
            st.regions_s,
            self._mode,
            self._active_segment(st.region_index) if sequence else -1,
            self._zoom_win_start,
            self._zoom_win_end,
        )
        if key != self._overlay_key:
            self._overlay_key = key
            self._draw_overlays(st.regions_s, st.region_index)

        # 4. Update Status text
        mins = int(st.position_s // 60)
        secs = st.position_s % 60
        time_str = f"{mins:02d}:{secs:06.3f}"

        beat_idx = self.song_map.beat_index_at(st.position_s)
        beat_num = self.song_map.beat_numbers[beat_idx] if len(self.song_map.beat_numbers) > 0 else 1
        bar_idx = self.song_map.bar_index_at(st.position_s)
        bar_str = f"bar {bar_idx + 1}.{beat_num}"
        beat_str = f"beat {beat_num}"

        p = self.song_map.phrase_at(st.position_s)
        phrase_str = f"phrase {p.index}/{len(self.song_map.phrases)} {p.label}" if p else "phrase --"

        seg_i = self._active_segment(st.region_index) if sequence else None
        if seg_i is not None and self._live_state is not None:
            seg = self._live_state.segments[seg_i]
            loop_str = f"seg {seg_i + 1}/{len(self._live_state.segments)} {seg.label}"
        elif loop_on and st.loop_end_s > st.loop_start_s:
            l_start_bar = self.song_map.bar_index_at(st.loop_start_s) + 1
            l_end_bar = self.song_map.bar_index_at(st.loop_end_s) + 1
            loop_str = f"loop {l_start_bar}–{l_end_bar}"
        else:
            loop_str = "loop off"

        self.status_lbl.config(
            text=f"{time_str}   {bar_str}   {beat_str}   {phrase_str}   {loop_str}"
        )

        self.root.after(33, self._tick)

    def run(self) -> None:
        if self.live is not None:
            self._apply_live(self.live.state)
        self._update_zoom_lane(0.0, force_redraw=True)
        self._focus_lanes()
        self.root.after(33, self._tick)
        self.root.mainloop()
