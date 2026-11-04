"""Main launcher for conga_player."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure this directory is at the front of sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pedalboard.io import AudioStream
from conga_player.live import LiveSession, LiveSyntaxError, format_time, parse_live
from conga_player.songmap import GRID_SOURCES, load_song


def find_default_song_folder(songs_root: Path) -> Path:
    if not songs_root.exists() or not songs_root.is_dir():
        raise FileNotFoundError(f"Songs directory not found: {songs_root}")

    subdirs = [p for p in songs_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No songs found in {songs_root}")
    if len(subdirs) == 1:
        return subdirs[0]

    # Multiple found without song arg
    options = "\n".join(f"  - {d.name}" for d in subdirs)
    raise ValueError(
        f"Multiple songs found in {songs_root}. Please specify one:\n{options}"
    )


def resolve_song_folder(song_arg: str | None, base_dir: Path) -> Path:
    songs_root = base_dir / "songs"
    if not song_arg:
        return find_default_song_folder(songs_root)

    cand = Path(song_arg)
    if cand.is_dir():
        return cand.resolve()

    # Try relative to songs_root
    in_songs = songs_root / song_arg
    if in_songs.is_dir():
        return in_songs.resolve()

    raise FileNotFoundError(f"Could not find song folder '{song_arg}'")


def resolve_device(device_arg: str | None) -> str | None:
    if not device_arg:
        return None

    devs = AudioStream.output_device_names
    # If passed as numeric index, e.g. "0" or "1"
    if device_arg.strip().isdigit():
        idx = int(device_arg.strip())
        if 0 <= idx < len(devs):
            return devs[idx]
        raise ValueError(f"Device index {idx} out of range (0 to {len(devs) - 1})")

    # Exact match
    if device_arg in devs:
        return device_arg

    # Case-insensitive substring match
    matches = [d for d in devs if device_arg.lower() in d.lower()]
    if matches:
        return matches[0]

    return device_arg


def list_devices() -> None:
    print("Available Output Audio Devices:")
    devs = AudioStream.output_device_names
    default_dev = AudioStream.default_output_device_name
    for i, dev in enumerate(devs):
        is_def = " (Default)" if dev == default_dev else ""
        print(f"  [{i}] {dev}{is_def}")


def run_selftest(song_folder: Path, live_path: Path | None = None, grid: str = "auto") -> None:
    song_map = load_song(song_folder, grid=grid)
    print(f"Grid source: {song_map.grid_source}")
    print(
        f"{len(song_map.beat_times_s)} beats / {len(song_map.bar_times_s)} bars / "
        f"{len(song_map.phrases)} phrases / {song_map.bpm:.1f} BPM / {song_map.duration_s:.2f} s"
    )
    if song_map.phrases:
        print(f"First phrase: {song_map.phrases[0].label}")
        print(f"Last phrase: {song_map.phrases[-1].label}")
    print(f"Audio path: {song_map.audio_path}")

    if live_path is not None:
        session = LiveSession(live_path, song_map)
        print(f"live.toml: {live_path}")
        try:
            state = parse_live(session.disk_text, song_map)
        except LiveSyntaxError as exc:
            print(f"  syntax error: {exc.error}")
            return
        print(f"  mode={state.mode} playing={state.playing} volume_db={state.volume_db}")
        if state.loop:
            print(f"  loop {state.loop.label} {format_time(state.loop.start_s)}\u2013{format_time(state.loop.end_s)}")
        if not state.segments:
            print("  (no segments)")
        for i, seg in enumerate(state.segments, start=1):
            rep_s = f" x{seg.repeat}" if seg.repeat > 1 else ""
            print(f"  [{i}] {seg.label} {format_time(seg.start_s)}\u2013{format_time(seg.end_s)}{rep_s}")
        for msg in state.errors:
            print(f"  error: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="conga_player — beat/phrase-mapped audio player")
    parser.add_argument("song", nargs="?", default=None, help="Song folder path or slug inside songs/")
    parser.add_argument("--list-devices", action="store_true", help="List audio output devices and exit")
    parser.add_argument("--device", type=str, default=None, help="Output audio device name")
    parser.add_argument("--buffer-size", type=int, default=2048, help="Audio buffer size in frames (default 2048)")
    parser.add_argument("--latency-offset-ms", type=float, default=None, help="Latency offset in ms for playhead position")
    parser.add_argument("--selftest", action="store_true", help="Load song map, print summary, and exit")
    parser.add_argument("--live", type=Path, default=None, help="Path to the live .toml session (default <song>/live.toml)")
    parser.add_argument("--no-live", action="store_true", help="Disable the live .toml session")
    parser.add_argument(
        "--grid",
        choices=GRID_SOURCES,
        default="auto",
        help="Beat map source: 'rekordbox' (track_dump.json: beat grid + phrases), "
        "'beat_this' (beat_this/*.beats: beats/downbeats/bars only, no phrases), "
        "or 'auto' (rekordbox if available, else beat_this; default)",
    )

    args = parser.parse_args()

    # Phrase tables and segment labels contain en dashes; never crash on a legacy console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if args.list_devices:
        list_devices()
        return

    song_folder = resolve_song_folder(args.song, APP_DIR)

    live_path = None if args.no_live else (args.live or song_folder / "live.toml")

    if args.selftest:
        run_selftest(song_folder, live_path, args.grid)
        return

    # Load audio into memory
    print(f"Loading song map from {song_folder.name}...")
    song_map = load_song(song_folder, grid=args.grid)
    print(f"Beat map source: {song_map.grid_source}")
    print(f"Loading audio from {song_map.audio_path.name}...")

    from pedalboard.io import AudioFile
    with AudioFile(str(song_map.audio_path)) as f:
        audio = f.read(f.frames)
        sr = f.samplerate

    print(f"Preloaded {audio.shape[1]} frames ({audio.shape[1] / sr:.2f} s) at {sr} Hz.")

    device_name = resolve_device(args.device)
    if device_name:
        print(f"Using audio device: {device_name}")

    from conga_player.engine import Engine
    engine = Engine(
        audio=audio,
        sample_rate=sr,
        buffer_size=args.buffer_size,
        output_device_name=device_name,
        latency_offset_ms=args.latency_offset_ms,
    )
    engine.start()

    from conga_player.ui import CongaPlayerApp
    live = LiveSession(live_path, song_map) if live_path is not None else None
    if live is not None:
        print(f"Live session: {live_path}")
    app = CongaPlayerApp(song_map, engine, audio, live)
    try:
        app.run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
