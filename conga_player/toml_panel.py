"""In-app live.toml editor: apply-as-you-type, error highlighting and an external-change banner."""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Optional

from conga_player.live import LiveParseError, LiveSession, LiveState, LiveSyntaxError, edit_text, parse_live

APPLY_DELAY_MS = 300

_BG = "#0b1220"
_FG = "#e2e8f0"

_RE_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_RE_KEY = re.compile(r"[A-Za-z_][\w-]*(?=\s*=)")
_RE_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'[^\'\n]*\'')
_RE_COMMENT = re.compile(r"#[^\n]*")


class TomlPanel(tk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        session: LiveSession,
        on_state: Callable[[LiveState], None],
        on_escape: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent, bg="#0f172a")
        self.session = session
        self.on_state = on_state
        self.on_escape = on_escape
        self.error: Optional[str] = None  # message currently shown (syntax or field error)
        self._after_id: Optional[str] = None
        self._queued: dict[str, Any] = {}
        self._seen_revision = session.revision
        self._synced = session.disk_text  # last text the buffer and the file agreed on

        top = tk.Frame(self, bg="#0f172a")
        top.pack(fill=tk.X)
        self.title_lbl = tk.Label(
            top, text="live.toml", bg="#0f172a", fg="#38bdf8", font=("Segoe UI", 10, "bold"), anchor="w"
        )
        self.title_lbl.pack(side=tk.LEFT, padx=(8, 4))
        self.unapplied_lbl = tk.Label(top, text="", bg="#0f172a", fg="#f59e0b", font=("Segoe UI", 9))
        self.unapplied_lbl.pack(side=tk.LEFT)
        tk.Label(
            top, text="Ctrl+Enter / Ctrl+S apply now   Esc back to player keys",
            bg="#0f172a", fg="#64748b", font=("Segoe UI", 8),
        ).pack(side=tk.RIGHT, padx=8)

        # Banner for outside changes while there are unapplied edits.
        self.banner = tk.Frame(self, bg="#78350f")
        tk.Label(
            self.banner, text="live.toml changed on disk", bg="#78350f", fg="#fef3c7", font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=8, pady=2)
        ttk.Button(self.banner, text="Reload", width=8, command=self._reload_from_disk).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.banner, text="Keep mine", width=10, command=self._keep_mine).pack(side=tk.LEFT, padx=2)

        body = tk.Frame(self, bg="#0f172a")
        body.pack(fill=tk.BOTH, expand=True, padx=(8, 8))
        self.text = tk.Text(
            body, undo=True, wrap="none", bg=_BG, fg=_FG, insertbackground="#f8fafc",
            selectbackground="#334155", font=("Consolas", 11), relief=tk.FLAT, borderwidth=0,
            padx=6, pady=4, height=10,
        )
        self.text.configure(maxundo=-1)
        yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=self.text.yview)
        self.text.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.error_lbl = tk.Label(
            self, text="", bg="#0f172a", fg="#fca5a5", font=("Consolas", 10), anchor="w", justify="left"
        )
        self.error_lbl.pack(fill=tk.X, padx=8, pady=(2, 4))

        self.text.tag_configure("number", foreground="#fbbf24")
        self.text.tag_configure("key", foreground="#7dd3fc")
        self.text.tag_configure("string", foreground="#86efac")
        self.text.tag_configure("comment", foreground="#64748b")
        self.text.tag_configure("errline", background="#7f1d1d")

        self.text.bind("<<Modified>>", self._on_modified)
        self.text.bind("<Control-Return>", self._apply_now_event)
        self.text.bind("<Control-s>", self._apply_now_event)
        self.text.bind("<Control-S>", self._apply_now_event)
        self.text.bind("<Escape>", self._on_escape)

        self._set_text(session.disk_text, keep_view=False, undoable=False)
        self.text.edit_reset()
        self._validate_display(self.text_value())
        self._update_marker()

    # ------------------------------------------------------------------ buffer helpers

    def text_value(self) -> str:
        return self.text.get("1.0", "end-1c")

    def has_unapplied(self) -> bool:
        return self.text_value() != self.session.text

    def _set_text(self, new: str, keep_view: bool = True, undoable: bool = True) -> None:
        """Replace the whole buffer (one undo step), keeping cursor line and scroll position."""
        if new == self.text_value():
            return
        cur_line, cur_col = (int(x) for x in self.text.index("insert").split("."))
        first_line = self.text.index("@0,0")
        # Tk would otherwise split delete+insert into two undo steps.
        self.text.configure(autoseparators=False)
        try:
            if undoable:
                self.text.edit_separator()
            self.text.delete("1.0", "end")
            self.text.insert("1.0", new)
            if undoable:
                self.text.edit_separator()
        finally:
            self.text.configure(autoseparators=True)
        if keep_view:
            last = int(self.text.index("end-1c").split(".")[0])
            self.text.mark_set("insert", f"{min(cur_line, last)}.{cur_col}")
            self.text.yview(first_line)
        self.text.edit_modified(False)
        self._highlight()

    def _highlight(self) -> None:
        for tag in ("number", "key", "string", "comment"):
            self.text.tag_remove(tag, "1.0", "end")
        content = self.text_value()
        for tag, rx in (("number", _RE_NUMBER), ("key", _RE_KEY), ("string", _RE_STRING), ("comment", _RE_COMMENT)):
            for m in rx.finditer(content):
                self.text.tag_add(tag, f"1.0+{m.start()}c", f"1.0+{m.end()}c")

    def _mark_error_lines(self, lines: list[int]) -> None:
        self.text.tag_remove("errline", "1.0", "end")
        for ln in lines:
            self.text.tag_add("errline", f"{ln}.0", f"{ln}.0 lineend+1c")

    def _update_marker(self) -> None:
        self.unapplied_lbl.config(text="● unapplied" if self.has_unapplied() else "")

    # ------------------------------------------------------------------ applying

    def _on_modified(self, _event: tk.Event) -> None:
        if not self.text.edit_modified():
            return
        self.text.edit_modified(False)
        self._highlight()
        self._update_marker()
        if self._after_id is not None:
            self.after_cancel(self._after_id)
        self._after_id = self.after(APPLY_DELAY_MS, self._check)

    def _apply_now_event(self, _event: tk.Event) -> str:
        self.apply_now()
        return "break"

    def apply_now(self) -> bool:
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        return self._try_apply(self.text_value())

    def _check(self) -> None:
        self._after_id = None
        self._try_apply(self.text_value())

    def _try_apply(self, text: str) -> bool:
        result = self.session.set_text(text)
        if isinstance(result, LiveParseError):
            self._show_error(str(result), [result.line] if result.line else [])
            self._update_marker()
            return False
        self._synced = text
        self._show_field_errors(result)
        self.on_state(result)
        self._update_marker()
        self._flush_queue()
        return True

    def _validate_display(self, text: str) -> None:
        """Show errors for text without applying or writing it."""
        try:
            state = parse_live(text, self.session.song_map)
        except LiveSyntaxError as exc:
            self._show_error(str(exc.error), [exc.error.line] if exc.error.line else [])
            return
        self._show_field_errors(state)

    def _show_field_errors(self, state: LiveState) -> None:
        if state.errors:
            self._show_error("\n".join(state.errors[:3]), [ln for ln in state.error_lines if ln])
        else:
            self._show_error(None, [])

    def _show_error(self, message: Optional[str], lines: list[int]) -> None:
        self.error = message
        self.error_lbl.config(text=message or "")
        self._mark_error_lines(lines)

    # ------------------------------------------------------------------ UI actions

    def apply_ui(self, **changes: Any) -> Optional[LiveState]:
        """Edit the panel text for a UI action (one undo step). Returns the applied state, if any."""
        if not self.has_unapplied():
            new = self.session.apply_ui(**changes)
            if new is None:
                self._queued.update(changes)
                return None
            self._synced = new
            self._set_text(new)
            self._update_marker()
            return self.session.state
        # The user has unapplied text: merge into it if it parses, otherwise queue.
        try:
            merged = edit_text(self.text_value(), **changes)
        except LiveSyntaxError:
            self._queued.update(changes)
            return None
        self._set_text(merged)
        if self.apply_now():
            return self.session.state
        return None

    def _flush_queue(self) -> None:
        if not self._queued:
            return
        queued, self._queued = self._queued, {}
        self.apply_ui(**queued)

    # ------------------------------------------------------------------ external changes

    def check_external(self) -> None:
        """Called from the UI poll: react to edits made outside the app."""
        if self.session.revision == self._seen_revision:
            return
        self._seen_revision = self.session.revision
        if self.text_value() != self._synced:
            # Unapplied typing: never clobber it silently.
            self.banner.pack(fill=tk.X, before=self.text.master)
            return
        self._reload_from_disk()

    def _reload_from_disk(self) -> None:
        self.banner.pack_forget()
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._synced = self.session.disk_text
        self._set_text(self.session.disk_text)
        self._validate_display(self.text_value())
        self._update_marker()

    def _keep_mine(self) -> None:
        self.banner.pack_forget()
        self.apply_now()

    # ------------------------------------------------------------------ focus

    def _on_escape(self, _event: tk.Event) -> str:
        if self.on_escape:
            self.on_escape()
        return "break"
