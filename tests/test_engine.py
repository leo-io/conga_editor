"""Unit tests for the Audio Engine without requiring an AudioStream device."""

import numpy as np
import pytest

from conga_player.engine import Engine


def test_paused_output_is_all_zeros():
    audio = np.ones((1, 100), dtype=np.float32)
    engine = Engine(audio, sample_rate=1000)
    assert not engine.state().playing

    block = engine._next_block(10)
    assert np.all(block == 0.0)


def test_seek_lands_on_exact_frame():
    audio = np.arange(100, dtype=np.float32).reshape(1, 100)
    engine = Engine(audio, sample_rate=1000, declick_ms=0.0)
    engine.seek(0.042)  # frame 42
    engine.play()

    block = engine._next_block(5)
    assert np.array_equal(block[0], [42, 43, 44, 45, 46])


def test_block_straddling_loop_end():
    # Audio with sequential frames
    audio = np.arange(100, dtype=np.float32).reshape(1, 100)
    engine = Engine(audio, sample_rate=1000, declick_ms=0.0)

    # Loop between frame 20 and 40 (length 20 frames)
    engine.set_loop(0.020, 0.040)
    engine.seek(0.035)  # 5 frames before loop end
    engine.play()

    block = engine._next_block(10)
    expected = np.array([35, 36, 37, 38, 39, 20, 21, 22, 23, 24], dtype=np.float32)
    assert np.array_equal(block[0], expected)


def test_loop_shorter_than_one_block_wraps_twice():
    audio = np.arange(100, dtype=np.float32).reshape(1, 100)
    engine = Engine(audio, sample_rate=1000, declick_ms=0.0)

    # Loop between frame 10 and 14 (length 4 frames)
    engine.set_loop(0.010, 0.014)
    engine.seek(0.010)
    engine.play()

    block = engine._next_block(10)
    expected = np.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11], dtype=np.float32)
    assert np.array_equal(block[0], expected)


def test_declick_ramps_applied_only_at_seams():
    # Constant 1.0 signal
    audio = np.ones((1, 100), dtype=np.float32)
    # declick_ms=2.0 at 1000 Hz = 2 frames
    engine = Engine(audio, sample_rate=1000, declick_ms=2.0)

    # Loop between frame 10 and 20
    engine.set_loop(0.010, 0.020)
    engine.seek(0.016)  # 4 frames before seam at index 4
    engine.play()

    block = engine._next_block(10)

    # Unseamed frames before fade out: index 0 and 1
    assert block[0, 0] == pytest.approx(1.0)
    assert block[0, 1] == pytest.approx(1.0)

    # Fade out at seam (indices 2, 3)
    assert block[0, 2] < 1.0
    assert block[0, 3] < 1.0

    # Fade in after seam (indices 4, 5)
    assert block[0, 4] < 1.0
    assert block[0, 5] < 1.0

    # Unseamed frames after fade in: indices 6, 7, 8, 9
    assert block[0, 6] == pytest.approx(1.0)
    assert block[0, 7] == pytest.approx(1.0)
    assert block[0, 8] == pytest.approx(1.0)
    assert block[0, 9] == pytest.approx(1.0)


def test_non_looping_playback_stops_at_end():
    audio = np.arange(20, dtype=np.float32).reshape(1, 20)
    engine = Engine(audio, sample_rate=1000, declick_ms=0.0)
    engine.seek(0.015)  # frame 15, only 5 frames left in audio
    engine.play()

    block = engine._next_block(10)
    assert np.array_equal(block[0, :5], [15, 16, 17, 18, 19])
    assert np.all(block[0, 5:] == 0.0)
    assert not engine.state().playing


# --- Region sequence tests ---


def _seq_engine(declick_ms=0.0, frames=200):
    audio = np.arange(frames, dtype=np.float32).reshape(1, frames)
    return Engine(audio, sample_rate=1000, declick_ms=declick_ms)


def test_three_regions_play_in_order_and_wrap():
    engine = _seq_engine()
    engine.set_sequence([(0.010, 0.013), (0.050, 0.052), (0.100, 0.104)])
    engine.seek(0.010)
    engine.play()

    block = engine._next_block(12)
    expected = [10, 11, 12, 50, 51, 100, 101, 102, 103, 10, 11, 12]
    assert np.array_equal(block[0], expected)


def test_repeat_expansion_plays_region_twice():
    engine = _seq_engine()
    engine.set_sequence([(0.010, 0.012), (0.010, 0.012), (0.030, 0.032)])
    engine.seek(0.010)
    engine.play()

    block = engine._next_block(8)
    assert np.array_equal(block[0], [10, 11, 10, 11, 30, 31, 10, 11])


def test_block_straddling_region_boundary():
    engine = _seq_engine()
    engine.set_sequence([(0.010, 0.014), (0.050, 0.060)])
    engine.seek(0.012)
    engine.play()

    block = engine._next_block(5)
    assert np.array_equal(block[0], [12, 13, 50, 51, 52])


def test_several_seams_in_one_block_are_declicked():
    audio = np.ones((1, 200), dtype=np.float32)
    engine = Engine(audio, sample_rate=1000, declick_ms=2.0)
    engine.set_sequence([(0.010, 0.016), (0.050, 0.056), (0.100, 0.106)])
    engine.seek(0.010)
    engine.play()

    block = engine._next_block(18)[0]
    for seam in (6, 12):
        assert block[seam - 1] < 1.0 and block[seam] < 1.0
    assert block[3] == pytest.approx(1.0) and block[9] == pytest.approx(1.0)


def test_set_sequence_inside_new_region_keeps_position():
    engine = _seq_engine()
    engine.seek(0.055)
    engine.set_sequence([(0.010, 0.020), (0.050, 0.060)])
    engine.play()

    block = engine._next_block(7)
    assert np.array_equal(block[0], [55, 56, 57, 58, 59, 10, 11])


def test_set_sequence_outside_every_region_jumps_to_first():
    engine = _seq_engine()
    engine.seek(0.150)
    engine.set_sequence([(0.010, 0.020), (0.050, 0.060)])
    engine.play()

    block = engine._next_block(3)
    assert block[0, 1] == 11 and block[0, 2] == 12
    assert engine.state().region_count == 2


def test_empty_sequence_is_free_play():
    engine = _seq_engine()
    engine.set_sequence([(0.010, 0.020)])
    engine.set_sequence([])
    assert not engine.state().loop_enabled


def test_playhead_maps_across_seam(monkeypatch):
    import conga_player.engine as eng

    clock = [100.0]
    monkeypatch.setattr(eng.time, "monotonic", lambda: clock[0])

    audio = np.arange(200, dtype=np.float32).reshape(1, 200)
    engine = Engine(audio, sample_rate=1000, declick_ms=0.0, latency_offset_ms=0.0)
    engine.set_sequence([(0.010, 0.014), (0.050, 0.060)])
    engine.seek(0.010)
    engine.play()

    clock[0] = 100.010
    engine._next_block(10)  # frames 10-13, then 50-55; block starts at t=100.010

    # 1 ms into the block -> frame 11; 5 ms in -> 51 (after the seam)
    clock[0] = 100.011
    assert engine.state().position_s == pytest.approx(0.011)
    assert engine.state().region_index == 0
    clock[0] = 100.015
    st = engine.state()
    assert st.position_s == pytest.approx(0.051)
    assert st.region_index == 1
