# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`conga_player` — a tkinter desktop player that opens a song folder analyzed in rekordbox and lets you jump to and loop any beat, bar or phrase, for practising congas over a track. Audio I/O goes through `pedalboard`. The repo directory is `conga_editor`; the package inside is `conga_player`, extracted from a larger `pedalboard` monorepo.

## Commands

Windows / PowerShell, with a repo-local venv (`requirements.txt`: `pedalboard==0.9.26`, `numpy`, `pytest`):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

.\.venv\Scripts\python main.py --selftest        # parse the song map, print summary, no audio device, no GUI
.\.venv\Scripts\python main.py --list-devices    # confirm a usable output device before debugging the GUI
.\.venv\Scripts\python main.py                   # launch (defaults to the only folder in songs/)
.\.venv\Scripts\python main.py songs\<slug> --device "<name>" --latency-offset-ms 50

.\.venv\Scripts\python -m pytest tests -q
.\.venv\Scripts\python -m pytest tests/test_engine.py::test_block_straddling_loop_end -q   # single test
```

`--selftest` is the cheapest check after touching `songmap.py`: it catches every map-parsing error without opening an audio device.

There is no pytest config file — `tests/conftest.py` puts the repo root on `sys.path` so `conga_player` imports resolve; run pytest from the repo root.

## Architecture

Strictly layered, one responsibility per module, no back-edges:

- **`conga_player/songmap.py`** — the *only* module that knows rekordbox's layout. Reads `track.json` (metadata, audio filename) and `_temp/track_dump.json` (beat grid, song structure) and returns a `SongMap`: beat/bar time arrays, `Phrase` list, 3-band preview bands. Beat times come from `anlz.DAT.beat_grid.beats[].time` (ms) **plus** `anlz.EXT.extended_beat_grid.beats[].microseconds` for sub-ms accuracy. Phrase labels depend on the track's `mood` (1/2/3 → different kind→name tables in `MOOD_LABELS`). Also owns the query helpers the UI uses: `beat_index_at`, `bar_index_at`, `phrase_at`, `snap(t, mode)`.
- **`conga_player/peaks.py`** — pure numpy waveform math: resample the stored 1200-column preview to canvas width, compute max-abs peak columns for the zoom lane, and synthesize fallback preview bands when the preview JSON is absent.
- **`conga_player/engine.py`** — playback. Knows nothing about beats, bars or phrases; its whole API is in **seconds**. Holds the entire decoded song as one `float32` `(channels, frames)` array (≈235 MB for the sample song) so seeks and loop wraps are instant.
- **`conga_player/ui.py`** — tkinter. Four fixed-width (1200px) canvas lanes: phrase blocks, 3-band overview, bar ruler, zoom lane (4/8/16/32 bars). Polls `engine.state()` from `_tick()` every 33 ms (~30 fps) and only moves canvas item coords; the zoom lane is redrawn only when the centre bar or zoom level changes.
- **`main.py`** — argparse, song-folder and device resolution, audio preload, wiring. Imports `engine`/`ui` lazily so `--selftest` and `--list-devices` never touch audio or Tk.

**Units.** The map layer is in seconds throughout; frames are derived at the engine's *actual* output rate. The device may refuse 44100, in which case `Engine._feed_loop` overwrites `self.sample_rate` from the opened stream — so never cache a frame count across engine start.

### Engine constraints (all verified against `pedalboard` 0.9.26 internals)

- Output-only `AudioStream` **silently ignores `stream.plugins`**, so the `Pedalboard` gain chain is applied in Python per block inside `_next_block`.
- `stream.write()` blocks on enqueue by **busy-spinning**, and the play FIFO is only `buffer_size + 1` frames deep. `_feed_loop` therefore paces itself with a `time.monotonic()` deadline (`_next_write`) and sleeps to ~2 ms before it. Low CPU while playing is the test that this is still correct — a pegged core means the sleep maths broke.
- Always use the `AudioStream` context manager; calling `write()` on a stopped stream reopens and closes the device per call.
- There is no queryable FIFO depth or underrun counter, so position is tracked in Python: `state()` interpolates from the last block write using the monotonic clock, minus a latency offset (`--latency-offset-ms`, defaulting to `buffer_size` frames). This is an estimate by construction — the playhead drifting early or late is tuned with that flag, not fixed in code.
- `_next_block(n)` is lock-protected and side-effect-free apart from engine state: the engine tests drive it directly with a synthetic ramp array and **no audio device**. Keep it that way.
- Loop seams get a short declick fade (`declick_ms`, 2 ms default) on both sides of every wrap within a block; a loop shorter than one block wraps several times per block and gets a fade per seam.

## Songs

A song folder (`songs/<title_slug>/`) needs `track.json`, a `track_dump.json` (looked up at `_temp/track_dump.json` first, then the folder root), and the audio file. `_temp/` is rekordbox-dump scratch: if it is regenerated or wiped the player stops working, and the coupling is confined to `songmap.py` so an ingest step writing a stable map could be slotted in there.

Audio files (`*.flac|wav|mp3|m4a`) are git-ignored for size. Both `--selftest` and `tests/test_songmap.py` call `load_song`, which **raises `FileNotFoundError` unless an audio file is physically present** in the folder — on a fresh clone those fail until the audio is restored. `test_songmap.py` also asserts exact values for the one checked-in song (1332 beats, 333 bars, 47 phrases, 120.1 BPM, 665.71 s); treat those as the regression fixture rather than editing them to match new output.

## Conventions

- `from __future__ import annotations` in every module; dataclasses for state (`Phrase` frozen, `SongMap`/`EngineState` plain); numpy arrays typed by shape in a trailing comment.
- All mutable engine state is guarded by `self._lock` and mutated only inside `with self._lock:` blocks.
- `.gitattributes` normalizes all text to LF; audio and `.bin` are binary. Keep new text extensions listed there.
- Out of scope for v1 (don't add unasked): metronome, count-in, time-stretch, stems, MIDI, playlists, resizable window.
