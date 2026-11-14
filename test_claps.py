"""Offline tests for claps (no microphone). Run:
uv run --with-requirements requirements.txt --with pytest pytest -q
"""

import sys
import types

import pytest

import claps as c

STEP = 0.016  # one loudness reading


def play(detector: c.ClapDetector, sounds, until: float, quiet: float = 0.01) -> list[float]:
    """Feed readings up to `until`. sounds = [(start, length, level)]. Returns the fire times."""
    fired = []
    for i in range(round(until / STEP)):
        now = i * STEP
        level = next((lvl for start, length, lvl in sounds if start <= now < start + length), quiet)
        if detector.feed(level, now):
            fired.append(now)
    return fired


def clap(at: float, level: float = 0.8) -> tuple[float, float, float]:
    return (at, 0.05, level)


def test_two_claps_fire_once_after_the_confirm_wait():
    fired = play(c.ClapDetector(), [clap(1.0), clap(1.3)], until=4.0)
    assert len(fired) == 1
    assert 1.3 + c.CLAP_CONFIRM_SECONDS <= fired[0] < 1.3 + c.CLAP_CONFIRM_SECONDS + 0.1


def test_one_clap_does_nothing():
    assert play(c.ClapDetector(), [clap(1.0)], until=3.0) == []


def test_claps_too_far_apart_do_nothing():
    assert play(c.ClapDetector(), [clap(1.0), clap(2.0)], until=4.0) == []


def test_a_late_second_clap_can_start_a_new_pair():
    assert len(play(c.ClapDetector(), [clap(1.0), clap(2.0), clap(2.3)], until=4.0)) == 1


def test_applause_is_not_a_double_clap():
    sounds = [clap(1.0 + 0.25 * i) for i in range(10)]
    assert play(c.ClapDetector(), sounds, until=6.0) == []


def test_quiet_taps_are_ignored():
    assert play(c.ClapDetector(), [clap(1.0, 0.1), clap(1.3, 0.1)], until=3.0) == []


def test_long_loud_sounds_are_ignored():
    talk = [(1.0, 0.5, 0.8), (1.7, 0.5, 0.8)]
    assert play(c.ClapDetector(), talk, until=4.0) == []


def test_claps_must_stand_out_from_a_noisy_room():
    detector = c.ClapDetector()
    assert play(detector, [clap(3.0, 0.3), clap(3.3, 0.3)], until=5.0, quiet=0.15) == []  # 2x the noise
    assert len(play(c.ClapDetector(), [clap(3.0, 0.9), clap(3.3, 0.9)], until=5.0, quiet=0.15)) == 1


def test_two_double_claps_fire_twice():
    sounds = [clap(1.0), clap(1.3), clap(4.0), clap(4.3)]
    assert len(play(c.ClapDetector(), sounds, until=6.0)) == 2


class FakeStream:
    def __init__(self, **kwargs) -> None:
        self.kwargs, self.running, self.closed = kwargs, False, False

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def close(self) -> None:
        self.closed = True


def fake_sounddevice(monkeypatch, stream=FakeStream):
    module = types.SimpleNamespace(InputStream=stream)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def test_listener_counts_double_claps_between_polls(monkeypatch):
    fake_sounddevice(monkeypatch)
    listener = c.ClapListener()
    assert listener._stream.running and listener._stream.kwargs["channels"] == 1
    for i in range(round(3.0 / STEP)):
        now = i * STEP
        listener.hear(0.8 if 1.0 <= now < 1.05 or 1.3 <= now < 1.35 else 0.01, now)
    assert listener.poll() == 1
    assert listener.poll() == 0
    listener.close()
    assert listener._stream.closed


def test_listener_explains_a_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # makes the import fail
    with pytest.raises(c.ClapError, match="requirements-claps.txt"):
        c.ClapListener()


def test_listener_explains_a_missing_microphone(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("no input device")

    fake_sounddevice(monkeypatch, broken)
    with pytest.raises(c.ClapError, match="microphone"):
        c.ClapListener()


# ------------------------------------------------------------ motion_detect


import motion_detect as m  # noqa: E402
from test_motion_detect import FakeCapture, fast_loop, moving_clip, no_gui  # noqa: E402, F401


class FakeClaps:
    def __init__(self, counts) -> None:
        self.counts, self.closed = list(counts), False

    def poll(self) -> int:
        return self.counts.pop(0) if self.counts else 0

    def close(self) -> None:
        self.closed = True


def test_claps_flag_is_off_by_default():
    assert m.parse_args([]).claps is False
    assert m.parse_args(["--claps"]).claps is True


def test_run_logs_double_claps_and_closes_the_listener(monkeypatch, capsys, fast_loop):  # noqa: F811
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    listener = FakeClaps([0, 1, 0, 1])
    m.run(0, govee_enabled=False, max_seconds=None, headless=True, claps=listener)
    assert capsys.readouterr().out.count("CLAP ") == 2
    assert listener.closed


def test_main_reports_a_clap_setup_problem(monkeypatch, capsys):
    def broken():
        raise c.ClapError("no microphone here")

    monkeypatch.setattr(m, "ClapListener", broken)
    assert m.main(["--claps", "--no-govee"]) == 2
    assert "no microphone here" in capsys.readouterr().err
