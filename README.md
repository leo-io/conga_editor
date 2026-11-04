# conga_player

Beat- and phrase-mapped desktop audio player built on `pedalboard` for practising congas and percussion over tracks analyzed in rekordbox.

## Features

- **Direct rekordbox dump ingestion**: Reads `track.json` and `_temp/track_dump.json` directly from the song directory.
- **Beat & Bar accurate navigation**: 3-lane display:
  - Phrase lane (mood-aware colored phrase blocks) with loop/segment shading and playhead
  - Bar ruler with 16-bar markers
  - Zoom lane (4, 8, 16, or 32 bars) centered on the playhead with peak waveform and beat gridlines
- **Instant looping & seeking**: Sample-exact seamless looping with automated declick fades at seam transitions.
- **Low CPU audio feeder**: Output-only `pedalboard.io.AudioStream` with monotonic-clock feeder pacing and streaming gain processing.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

> **Note:** song audio files (`.flac`/`.wav`/`.mp3`) are git-ignored because of their size.
> Place the audio alongside `track.json` inside the song folder before running.

## Running

```powershell
# Run selftest (verifies parsing without opening audio devices or GUI)
.\.venv\Scripts\python main.py --selftest

# List available audio output devices
.\.venv\Scripts\python main.py --list-devices

# Launch player with the default song
.\.venv\Scripts\python main.py

# Launch player with a specific song folder or audio device
.\.venv\Scripts\python main.py songs\i_want_you_she_s_so_heavy_eric_kupper --device "Driver de som primário"
```

## Flags

- `--list-devices`: List all available audio output devices and exit.
- `--device NAME`: Specify audio output device name (defaults to system default).
- `--buffer-size FRAMES`: Audio stream buffer size in frames (default: 2048).
- `--latency-offset-ms MS`: Override playhead latency compensation in milliseconds.
- `--selftest`: Load song map, print summary statistics, and exit without opening audio or UI.
- `--live PATH`: Use this live `.toml` session file (default: `<song>/live.toml`).
- `--no-live`: Disable the live `.toml` session (single in-memory loop, nothing persisted).
- `--grid {auto,rekordbox,beat_this}`: Beat map source. `rekordbox` reads `track_dump.json` (beat grid + phrases); `beat_this` reads `<song>/beat_this/*.beats` (`<time_s> <beat_number>` per line, `1` = downbeat → beats, downbeats and bars only, no phrases). `auto` (default) uses rekordbox when `track_dump.json` exists, otherwise beat_this. Key, title and the waveform preview still come from the rekordbox files when present.

## Keyboard & Mouse Controls

| Input | Effect |
|---|---|
| **Space** | Toggle play / pause |
| **S** | Stop playback and rewind to beginning |
| **L** | Toggle `mode` between `"loop"` and `"free"` |
| **Q** / **Seq** button | Switch to `mode = "sequence"` (play the `segments` list) |
| **Esc** | Free play: removes `loop` (in the TOML panel: return focus to the player keys) |
| **P** | Loop currently playing phrase |
| **1**, **2**, **4**, **8** | Loop 1, 2, 4, or 8 bars starting from current bar downbeat |
| **[** / **]** | Jump to previous / next phrase start |
| **←** / **→** | Seek ∓1 bar |
| **Shift + ←** / **Shift + →** | Seek ∓1 beat |
| **-** / **+** | Zoom lane 4 / 8 / 16 / 32 bars |
| **Click zoom** | Seek to position (snapped according to snap mode) |
| **Drag zoom** | Set loop region from drag start to drag end and enable loop |
| **Click phrase** | Seek to phrase start |
| **Double-click phrase** | Loop phrase |

## Live session: `live.toml`

Every song folder gets a human-readable `live.toml` that always describes what the player is doing.
It is created from the song map on first run (including a commented phrase table to copy from).
The window has a **TOML panel** below the lanes (drag the divider to resize):

- Valid changes apply about 300 ms after you stop typing (**Ctrl+Enter** / **Ctrl+S** apply at once).
  Every applied change is saved to disk.
- An invalid edit highlights its line in red with a message; the player keeps the last good state.
  A bad segment is skipped and the others still play.
- UI actions (P, 1/2/4/8, drag, volume, play/stop, L, Q, Esc) edit the panel text in place, keeping
  your comments; Ctrl+Z in the panel undoes them. Seeks are not written.
- Editing the file in another editor works too: changes are picked up within ~200 ms. If you have
  unapplied typing in the panel, a banner offers **Reload** or **Keep mine**.
- While the panel has focus, player shortcuts are disabled; **Esc** returns to them.

```toml
mode = "sequence"      # "sequence" | "loop" | "free"
playing = true
volume_db = -3.0
loop = { bars = [17, 24] }        # used when mode = "loop"

segments = [                       # used when mode = "sequence"; plays in order, wraps
    [1, 32],                       # bare numbers are BEATS (1-based, inclusive end)
    { phrase = 8, repeat = 2 },
    { bars = [33, 36] },
    ["2m40s", "2m50s"],            # time strings: "2m40s", "02:40", "160s"
]
```

An endpoint that is a number is a beat; a string is a time, so `[65, "2m50s"]` is valid. Segments also
accept `{ beats = [a, b] }` and `{ from = ..., to = ... }`. Editing the file while playing keeps the
playhead if it is still inside one of the new segments, otherwise it jumps to the first segment.
The playhead position is not stored in the file.

## Running Tests

```powershell
.\.venv\Scripts\python -m pytest tests -q
```
