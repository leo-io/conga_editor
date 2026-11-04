"""Tests for songmap parsing and query helpers."""

from pathlib import Path

import numpy as np
import pytest

from conga_player.songmap import group_phrases, load_song

SONG_FOLDER = Path(__file__).resolve().parent.parent / "songs" / "conversations"
KID_FOLDER = Path(__file__).resolve().parent.parent / "songs" / "k_i_d" / "you_don_t_like_my_music"


@pytest.fixture(scope="module")
def song_map():
    assert SONG_FOLDER.is_dir(), f"Song folder missing: {SONG_FOLDER}"
    return load_song(SONG_FOLDER)


@pytest.fixture(scope="module")
def kid_song_map():
    if not KID_FOLDER.is_dir():
        pytest.skip(f"Song folder missing: {KID_FOLDER}")
    return load_song(KID_FOLDER)


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

    # First and last section labels (High mood: variant suffix; last label from the dump's derived sections)
    assert song_map.phrases[0].label == "Intro 1"
    assert song_map.phrases[-1].label == "Kind6"

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


def test_grouped_sections_counts(kid_song_map):
    assert len(kid_song_map.beat_times_s) == 682
    assert len(kid_song_map.bar_times_s) == 171
    # 25 raw phrases grouped into 16 sections
    assert len(kid_song_map.raw_phrases) == 25
    assert len(kid_song_map.sections) == 16
    assert len(kid_song_map.phrases) == 16
    assert kid_song_map.has_grouped_sections is True


def test_grouped_sections_sequential_numbering(kid_song_map):
    sections = kid_song_map.sections
    # Sequential 1..N indices
    assert [s.index for s in sections] == list(range(1, 17))

    # Downbeat alignment
    for s in sections:
        b_idx = s.start_beat - 1
        assert kid_song_map.beat_numbers[b_idx] == 1, f"Section {s.index} does not start on downbeat"

    # Check first section
    assert sections[0].index == 1
    assert sections[0].start_phrase_index == 1
    assert sections[0].phrase_count == 1
    assert sections[0].label == "Intro"
    assert sections[0].phrase_range_str == "phrase 1"

    # Check multi-phrase section (section 3: phrases 3-5)
    assert sections[2].index == 3
    assert sections[2].start_phrase_index == 3
    assert sections[2].phrase_count == 3
    assert sections[2].end_phrase_index == 5
    assert sections[2].phrase_range_str == "phrases 3-5"

    # Check contiguous boundaries
    for i in range(len(sections) - 1):
        assert sections[i].end_beat == sections[i + 1].start_beat
        assert sections[i].end_s == pytest.approx(sections[i + 1].start_s, abs=1e-5)


def test_snap_section(kid_song_map):
    s0 = kid_song_map.sections[0]
    s1 = kid_song_map.sections[1]
    mid = (s0.start_s + s1.start_s) / 2.0
    assert kid_song_map.snap(mid - 0.1, "section") == pytest.approx(s0.start_s, abs=1e-5)
    assert kid_song_map.snap(mid + 0.1, "section") == pytest.approx(s1.start_s, abs=1e-5)
    # "phrase" mode also snaps to section boundaries
    assert kid_song_map.snap(mid - 0.1, "phrase") == pytest.approx(s0.start_s, abs=1e-5)


def test_group_phrases_utility():
    phrases = [
        {"beat": 1, "kind": 1, "index": 1, "k1": 0, "k2": 0, "k3": 0},
        {"beat": 17, "kind": 2, "index": 2, "k1": 0, "k2": 0, "k3": 0},
        {"beat": 33, "kind": 2, "index": 3, "k1": 0, "k2": 0, "k3": 0},
        {"beat": 49, "kind": 3, "index": 4, "k1": 0, "k2": 0, "k3": 0},
    ]
    grouped = group_phrases(phrases, mood=2, end_beat=65)
    assert len(grouped) == 3
    # Check sequential numbering 1..3
    assert [g["index"] for g in grouped] == [1, 2, 3]
    # Check start_phrase_index
    assert [g["start_phrase_index"] for g in grouped] == [1, 2, 4]
    # Check phrase_count
    assert [g["phrase_count"] for g in grouped] == [1, 2, 1]
    # Merged section 2 (phrases 2 and 3)
    assert grouped[1]["start_beat"] == 17
    assert grouped[1]["end_beat"] == 49
    assert grouped[1]["beats"] == 32
    assert grouped[1]["bars"] == 8.0
