"""Tests for live.toml parsing, resolving, editing and syncing (no dump, no audio)."""

import os
import numpy as np
import pytest

from conga_player.live import (
    LiveParseError,
    LiveSession,
    LiveSyntaxError,
    edit_text,
    loop_spec_for_range,
    parse_live,
    parse_time_val,
    starter_text,
)
from conga_player.songmap import Phrase, SongMap


@pytest.fixture
def song_map(tmp_path):
    # 40 beats, 0.5 s each (120 BPM), starting at 1.0 s; 10 bars; duration 22 s.
    beats = 1.0 + 0.5 * np.arange(40)
    nums = (np.arange(40) % 4 + 1).astype(np.int8)
    first = np.where(nums == 1)[0]
    phrases = [
        Phrase(1, 1, "Intro", 1, 17, float(beats[0]), float(beats[16]), 4.0),
        Phrase(2, 2, "Verse1", 17, 41, float(beats[16]), 22.0, 6.0),
    ]
    return SongMap(
        slug="t", artist="a", title="t", key="", audio_path=tmp_path / "x.wav", duration_s=22.0,
        sample_rate=44100, bpm=120.0, beat_times_s=beats, beat_numbers=nums,
        bar_times_s=beats[first], bar_first_beat=first, phrases=phrases,
        preview_bands=np.zeros((1200, 3), dtype=np.uint8),
    )


def parse(song_map, body):
    return parse_live(body, song_map)


def test_parse_time_val_forms():
    assert parse_time_val("2m40s") == 160.0
    assert parse_time_val("0m12.42s") == pytest.approx(12.42)
    assert parse_time_val("02:40") == 160.0
    assert parse_time_val("1:02:40") == 3760.0
    assert parse_time_val("160s") == 160.0
    assert parse_time_val("160.5") == 160.5
    with pytest.raises(ValueError):
        parse_time_val("abc")


def test_bare_pair_is_beats_with_inclusive_end(song_map):
    st = parse(song_map, "segments = [[1, 4]]")
    seg = st.segments[0]
    assert seg.start_s == pytest.approx(1.0)
    assert seg.end_s == pytest.approx(song_map.beat_times_s[4])  # beat 5's time
    assert seg.label == "beats 1\u20134"


def test_last_beat_ends_at_duration(song_map):
    st = parse(song_map, "segments = [[37, 40]]")
    assert st.segments[0].end_s == 22.0


def test_bars_phrase_mixed_and_repeat(song_map):
    st = parse(
        song_map,
        'segments = [{ bars = [2, 3] }, { phrase = 2, repeat = 3 }, [5, "0m12s"], { from = 3, to = "00:08" }]',
    )
    assert not st.errors
    bars = song_map.bar_times_s
    assert (st.segments[0].start_s, st.segments[0].end_s) == (bars[1], bars[3])
    assert st.segments[1].repeat == 3 and st.segments[1].start_s == song_map.phrases[1].start_s
    assert (st.segments[2].start_s, st.segments[2].end_s) == (song_map.beat_times_s[4], 12.0)
    assert (st.segments[3].start_s, st.segments[3].end_s) == (song_map.beat_times_s[2], 8.0)


def test_errors_have_position_and_others_still_resolve(song_map):
    text = "segments = [\n    [1, 4],\n    [30, 2000],\n    [9, 3],\n    [5, 8],\n]\n"
    st = parse(song_map, text)
    assert [s.label for s in st.segments] == ["beats 1\u20134", "beats 5\u20138"]
    assert st.errors[0].startswith("segments[1]: beat 2000 out of range (1–40)")
    assert "start must be before end" in st.errors[1]
    assert st.error_lines == [3, 4]


def test_bad_field_values_are_reported(song_map):
    st = parse(song_map, 'mode = "bogus"\nplaying = 3\nloop = { phrase = 9 }')
    assert len(st.errors) == 3 and st.mode == "free"


def test_syntax_error_raises_with_line(song_map):
    with pytest.raises(LiveSyntaxError) as exc:
        parse(song_map, 'playing = true\nmode = "seq')
    assert exc.value.error.line == 2


def test_starter_file_parses_cleanly(song_map):
    text = starter_text(song_map)
    st = parse(song_map, text)
    assert not st.errors and st.mode == "sequence" and not st.playing and len(st.segments) == len(song_map.phrases)
    assert "phrase  1  Intro" in text and "phrase  2  Verse1" in text


def test_edit_text_keeps_everything_outside_changed_keys(song_map):
    text = starter_text(song_map)
    new = edit_text(text, mode="loop", volume_db=-3.5, loop={"bars": [2, 3]})
    old_lines = text.split("\n")
    new_lines = new.split("\n")
    assert 'mode = "loop"  # "sequence" | "loop" | "free"' in new_lines
    assert "volume_db = -3.5" in new_lines
    assert "loop = { bars = [2, 3] }" in new_lines
    changed = [l for l in old_lines if l not in new_lines]
    assert sorted(changed) == sorted(['mode = "sequence"  # "sequence" | "loop" | "free"', "volume_db = 0.0"])
    assert [l for l in new_lines if l.startswith("# phrase")] == [l for l in old_lines if l.startswith("# phrase")]
    st = parse(song_map, new)
    assert st.mode == "loop" and st.loop.label == "bars 2\u20133"


def test_edit_text_removes_loop(song_map):
    text = edit_text(starter_text(song_map), loop=[1, 4])
    assert "loop = [1, 4]" in text
    assert "\nloop =" not in edit_text(text, loop=None)


def test_loop_spec_for_range(song_map):
    b = song_map.beat_times_s
    assert loop_spec_for_range(song_map, b[0], b[8], "beat") == [1, 8]
    assert loop_spec_for_range(song_map, b[36], 22.0, "beat") == [37, 40]
    assert loop_spec_for_range(song_map, 2.0, 3.25, "off") == ["0:02.000", "0:03.250"]


def test_write_then_poll_is_echo_suppressed(song_map, tmp_path):
    session = LiveSession(tmp_path / "live.toml", song_map)
    assert session.apply_ui(mode="loop", loop=[1, 4]) is not None
    assert session.poll() is None
    assert session.state.mode == "loop"
    assert "mode = \"loop\"" in (tmp_path / "live.toml").read_text(encoding="utf-8")


def test_external_edit_is_picked_up(song_map, tmp_path):
    path = tmp_path / "live.toml"
    session = LiveSession(path, song_map)
    rev = session.revision
    path.write_text('mode = "sequence"\nsegments = [[1, 4]]\n', encoding="utf-8")
    state = session.poll()
    assert state is not None and state.mode == "sequence" and len(state.segments) == 1
    assert session.revision == rev + 1


def test_broken_file_keeps_last_good_state_and_refuses_update(song_map, tmp_path):
    path = tmp_path / "live.toml"
    session = LiveSession(path, song_map)
    session.apply_ui(mode="sequence", segments=[[1, 4]])
    path.write_text('mode = "seq', encoding="utf-8")
    assert session.poll() is None
    assert isinstance(session.error, LiveParseError)
    assert session.state.mode == "sequence"
    assert session.update(volume_db=-1.0) is None
    assert path.read_text(encoding="utf-8") == 'mode = "seq'
    assert "UI change not saved" in session.message


def test_set_text_invalid_writes_nothing(song_map, tmp_path):
    path = tmp_path / "live.toml"
    session = LiveSession(path, song_map)
    before = path.read_bytes()
    assert isinstance(session.set_text('mode = "x'), LiveParseError)
    assert path.read_bytes() == before


def test_missing_file_is_created_from_song_map(song_map, tmp_path):
    path = tmp_path / "sub" / "live.toml"
    LiveSession(path, song_map)
    assert path.exists() and "phrase  1  Intro" in path.read_text(encoding="utf-8")
