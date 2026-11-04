"""Tests for songmap parsing and query helpers."""

from pathlib import Path
import numpy as np
import pytest

from conga_player.songmap import load_song

SONG_FOLDER = Path(__file__).resolve().parent.parent / "songs" / "conversations"


@pytest.fixture(scope="module")
def song_map():
    assert SONG_FOLDER.is_dir(), f"Song folder missing: {SONG_FOLDER}"
    return load_song(SONG_FOLDER)


def test_song_counts(song_map):
    assert len(song_map.beat_times_s) == 771
    assert len(song_map.bar_times_s) == 193
    assert len(song_map.phrases) == 9
    assert song_map.bpm == pytest.approx(112.0, abs=1e-3)
    assert song_map.duration_s == pytest.approx(413.04, abs=0.01)


def test_beat_times(song_map):
    assert np.all(np.diff(song_map.beat_times_s) > 0), "beat_times_s must be strictly increasing"
    assert song_map.beat_times_s[0] == pytest.approx(0.000861, abs=1e-5)
    assert song_map.beat_times_s[-1] == pytest.approx(412.500839, abs=1e-5)


def test_phrases(song_map):
    # Every phrase starts on a downbeat
    for p in song_map.phrases:
        beat_idx = p.start_beat - 1
        assert song_map.beat_numbers[beat_idx] == 1, f"Phrase {p.index} does not start on downbeat"

    # Phrase 0 and last phrase labels
    assert song_map.phrases[0].label == "Intro"
    assert song_map.phrases[-1].label == "Kind 6"

    # Spans are contiguous
    for i in range(len(song_map.phrases) - 1):
        assert song_map.phrases[i].end_beat == song_map.phrases[i + 1].start_beat
        assert song_map.phrases[i].end_s == pytest.approx(song_map.phrases[i + 1].start_s, abs=1e-5)


def test_snap_midpoint(song_map):
    # Test snap around a midpoint of two consecutive bars
    b0 = song_map.bar_times_s[10]
    b1 = song_map.bar_times_s[11]
    mid = (b0 + b1) / 2.0

    # Slightly to the left
    assert song_map.snap(mid - 0.05, "bar") == pytest.approx(b0, abs=1e-5)
    # Slightly to the right
    assert song_map.snap(mid + 0.05, "bar") == pytest.approx(b1, abs=1e-5)

    # Test snap off
    assert song_map.snap(12.345, "off") == pytest.approx(12.345, abs=1e-5)

    # Test snap beat
    bt0 = song_map.beat_times_s[20]
    bt1 = song_map.beat_times_s[21]
    bmid = (bt0 + bt1) / 2.0
    assert song_map.snap(bmid - 0.02, "beat") == pytest.approx(bt0, abs=1e-5)
    assert song_map.snap(bmid + 0.02, "beat") == pytest.approx(bt1, abs=1e-5)


def test_auto_grid_prefers_rekordbox(song_map):
    assert song_map.grid_source == "rekordbox"


def _beat_this_folder(tmp_path: Path, with_dump: bool) -> Path:
    folder = tmp_path / "song"
    (folder / "beat_this").mkdir(parents=True)
    (folder / "song.wav").write_bytes(b"")  # only needs to exist
    # 2 bars of 4, then a 3-beat bar, then a downbeat: 0.5 s per beat = 120 BPM
    nums = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 1]
    text = "\n".join(f"{0.25 + 0.5 * i:.2f}\t{n}" for i, n in enumerate(nums)) + "\n"
    (folder / "beat_this" / "song.beats").write_text(text, encoding="utf-8")
    (folder / "track.json").write_text(
        '{"artist": "A", "title": "T", "files": [{"path": "song.wav", "duration": 6.0}]}', encoding="utf-8"
    )
    if with_dump:
        dump = (SONG_FOLDER / "raw_analisys_data" / "track_dump.json").read_text(encoding="utf-8")
        (folder / "track_dump.json").write_text(dump, encoding="utf-8")
    preview = "[" + ",".join(['{"low": 1, "mid": 2, "high": 3}'] * 1200) + "]"
    (folder / "anlz.EX2.waveform_3band_preview.data.json").write_text(preview, encoding="utf-8")
    return folder


def test_beat_this_grid(tmp_path):
    folder = _beat_this_folder(tmp_path, with_dump=True)
    m = load_song(folder, grid="beat_this")
    assert m.grid_source == "beat_this"
    assert len(m.beat_times_s) == 12
    assert m.beat_times_s[0] == pytest.approx(0.25)
    assert m.bar_first_beat.tolist() == [0, 4, 8, 11]
    assert m.bar_times_s.tolist() == pytest.approx([0.25, 2.25, 4.25, 5.75])
    assert m.bpm == pytest.approx(120.0)
    assert m.phrases == []
    assert m.key  # metadata still read from track_dump.json
    assert m.snap(2.0, "bar") == pytest.approx(2.25)
    assert m.snap(2.0, "phrase") == pytest.approx(2.0)


def test_auto_grid_falls_back_to_beat_this(tmp_path):
    m = load_song(_beat_this_folder(tmp_path, with_dump=False))
    assert m.grid_source == "beat_this"
    assert m.title == "T"


def test_missing_grid_source_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_song(_beat_this_folder(tmp_path, with_dump=False), grid="rekordbox")
    with pytest.raises(FileNotFoundError):
        load_song(SONG_FOLDER, grid="beat_this")
    with pytest.raises(ValueError):
        load_song(SONG_FOLDER, grid="nope")
