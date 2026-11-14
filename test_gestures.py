"""Offline tests for gestures. Run:
uv run --with-requirements requirements.txt --with pytest pytest -q

The pure logic needs no MediaPipe. One test runs the real model; it is skipped
unless MediaPipe, the model file and the sample photos are present.
"""

from pathlib import Path

import numpy as np
import pytest

import gestures as g
import motion_detect as m
from test_motion_detect import FakeCapture, fast_loop, moving_clip, no_gui  # noqa: F401


def hand(gesture: str | None, score: float = 0.9, box=(0.4, 0.4, 0.2, 0.2)) -> g.Hand:
    return g.Hand(gesture, score, "Right", box)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class ScriptedRecognizer:
    """Returns one prepared list of hands per call, then no hands."""

    def __init__(self, script: list[list[g.Hand]]) -> None:
        self.script = list(script)
        self.calls = 0
        self.closed = False

    def recognize(self, frame, now: float) -> list[g.Hand]:
        self.calls += 1
        return self.script.pop(0) if self.script else []

    def close(self) -> None:
        self.closed = True


FRAME = np.zeros((72, 128, 3), np.uint8)


# ------------------------------------------------------------------ trigger


def test_trigger_fires_once_after_the_hold_time():
    trigger = g.HoldTrigger(hold=1.0, grace=0.3)
    assert trigger.update("Victory", 0.0) is None
    assert trigger.update("Victory", 0.5) is None
    assert trigger.update("Victory", 1.0) == "Victory"
    assert trigger.update("Victory", 1.5) is None  # still held: no repeat
    assert trigger.update("Victory", 9.0) is None


def test_trigger_needs_a_release_before_firing_again():
    trigger = g.HoldTrigger(hold=1.0, grace=0.3)
    trigger.update("Victory", 0.0)
    assert trigger.update("Victory", 1.0) == "Victory"
    assert trigger.update(None, 2.0) is None  # hand dropped
    assert trigger.update("Victory", 3.0) is None  # new hold starts here
    assert trigger.update("Victory", 4.0) == "Victory"


def test_trigger_survives_a_short_flicker():
    trigger = g.HoldTrigger(hold=1.0, grace=0.3)
    trigger.update("Victory", 0.0)
    trigger.update("Victory", 0.4)
    assert trigger.update(None, 0.5) is None  # one bad frame
    assert trigger.update("Pointing_Up", 0.6) is None  # misread for a moment
    assert trigger.update("Victory", 0.7) is None
    assert trigger.update("Victory", 1.0) == "Victory"


def test_trigger_restarts_when_the_gesture_really_changes():
    trigger = g.HoldTrigger(hold=1.0, grace=0.3)
    trigger.update("Victory", 0.0)
    trigger.update("Victory", 0.4)
    trigger.update("Closed_Fist", 0.5)  # inside the grace: could be a flicker
    assert trigger.update("Closed_Fist", 0.9) is None  # past the grace: fist takes over now
    assert trigger.update("Closed_Fist", 1.2) is None
    assert trigger.update("Closed_Fist", 2.0) == "Closed_Fist"


def test_trigger_with_no_hold_fires_at_once():
    assert g.HoldTrigger(hold=0.0, grace=0.3).update("Victory", 5.0) == "Victory"


def test_trigger_progress_fills_up_then_stays_full():
    trigger = g.HoldTrigger(hold=2.0, grace=0.3)
    assert trigger.progress(0.0) == 0.0
    trigger.update("Victory", 0.0)
    assert trigger.progress(1.0) == pytest.approx(0.5)
    trigger.update("Victory", 2.0)
    assert trigger.progress(5.0) == 1.0


# ------------------------------------------------------------------ picking


def test_best_hand_ignores_unknown_and_weak_gestures():
    hands = [hand(None, 0.99), hand("Victory", 0.4), hand("Open_Palm", 0.7), hand("Thumb_Up", 0.8)]
    assert g.best_hand(hands, min_score=0.6).gesture == "Thumb_Up"
    assert g.best_hand([hand(None), hand("Victory", 0.2)], min_score=0.6) is None
    assert g.best_hand([], min_score=0.6) is None


# ------------------------------------------------------------------- engine


def engine_with(script, clock, **kwargs) -> g.GestureEngine:
    return g.GestureEngine(ScriptedRecognizer(script), clock=clock, rate=0.0, **kwargs)


def hold(engine: g.GestureEngine, clock: FakeClock, seconds: float, step: float = 0.1) -> list[g.Event]:
    events = []
    for _ in range(round(seconds / step) + 1):
        events += engine.process(FRAME)
        clock.now += step
    return events


def test_engine_starts_armed_and_a_held_palm_disarms_it():
    clock = FakeClock()
    engine = engine_with([[hand("Open_Palm")]] * 30, clock, hold=1.0)
    assert engine.armed is True
    events = hold(engine, clock, 1.5)
    assert [(e.gesture, e.action) for e in events] == [("Open_Palm", "disarm")]
    assert engine.armed is False


def test_engine_fist_arms_again():
    clock = FakeClock()
    script = [[hand("Open_Palm")]] * 12 + [[]] * 5 + [[hand("Closed_Fist")]] * 12
    engine = engine_with(script, clock, hold=1.0)
    events = hold(engine, clock, 3.0)
    assert [e.action for e in events] == ["disarm", "arm"]
    assert engine.armed is True


def test_engine_reports_other_actions_without_changing_the_armed_state():
    clock = FakeClock()
    engine = engine_with([[hand("Victory")]] * 30, clock, hold=1.0, bindings={"Victory": "party"})
    events = hold(engine, clock, 1.5)
    assert [(e.gesture, e.action) for e in events] == [("Victory", "party")]
    assert engine.armed is True


def test_engine_ignores_gestures_with_no_binding():
    clock = FakeClock()
    engine = engine_with([[hand("ILoveYou")]] * 30, clock, hold=1.0, bindings={"Victory": "party"})
    assert hold(engine, clock, 1.5) == []


def test_engine_only_asks_the_model_at_the_configured_rate():
    clock = FakeClock()
    recognizer = ScriptedRecognizer([[hand("Victory")]] * 100)
    engine = g.GestureEngine(recognizer, clock=clock, rate=10.0)
    for _ in range(30):  # 30 frames in one second
        engine.process(FRAME)
        clock.now += 1 / 30
    assert 9 <= recognizer.calls <= 11
    assert engine.hands  # hands from the last look stay available for drawing


def test_engine_banner_expires():
    clock = FakeClock()
    engine = engine_with([[hand("Open_Palm")]] * 12, clock, hold=1.0)
    hold(engine, clock, 1.2)
    assert engine.banner() == "DISARMED by Open Palm"
    clock.now += g.BANNER_SECONDS + 1
    assert engine.banner() is None


def test_engine_close_closes_the_recognizer():
    recognizer = ScriptedRecognizer([])
    engine = g.GestureEngine(recognizer)
    engine.close()
    assert recognizer.closed and engine.closed


def test_pretty_gesture_names():
    assert g.pretty("Closed_Fist") == "Closed Fist"
    assert g.pretty("ILoveYou") == "I Love You"


# ------------------------------------------------------------------ drawing


def test_draw_gestures_marks_the_frame():
    clock = FakeClock()
    engine = engine_with([[hand("Victory")]], clock)
    engine.process(FRAME)
    frame = np.zeros((360, 640, 3), np.uint8)
    g.draw_gestures(frame, engine)
    assert frame.any()


# -------------------------------------------------------------------- model


def test_ensure_model_keeps_an_existing_file(tmp_path, monkeypatch):
    model = tmp_path / "model.task"
    model.write_bytes(b"x")
    monkeypatch.setattr(g, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    assert g.ensure_model(model) == model


def test_ensure_model_downloads_a_missing_file(tmp_path, monkeypatch, capsys):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"model-bytes"

    monkeypatch.setattr(g, "urlopen", lambda url, timeout: Response())
    model = tmp_path / "models" / "model.task"
    assert g.ensure_model(model).read_bytes() == b"model-bytes"
    assert "Downloading" in capsys.readouterr().out


def test_ensure_model_explains_a_failed_download(tmp_path, monkeypatch):
    def offline(url, timeout):
        raise OSError("no network")

    monkeypatch.setattr(g, "urlopen", offline)
    with pytest.raises(g.GestureError, match="no network"):
        g.ensure_model(tmp_path / "model.task")
    assert not list(tmp_path.iterdir())  # no half-written file left behind


SAMPLES = Path(__file__).parent / "models" / "samples"


@pytest.mark.skipif(
    not (g.MODEL_PATH.exists() and SAMPLES.exists()), reason="needs the downloaded model and sample photos"
)
@pytest.mark.parametrize(
    "photo, gesture",
    [("thumbs_up", "Thumb_Up"), ("thumbs_down", "Thumb_Down"), ("victory", "Victory"), ("pointing_up", "Pointing_Up")],
)
def test_real_model_recognizes_sample_photos(photo, gesture):
    pytest.importorskip("mediapipe")
    recognizer = g.MediaPipeRecognizer()
    try:
        frame = m.cv2.imread(str(SAMPLES / f"{photo}.jpg"))
        hands = [recognizer.recognize(frame, i / 30) for i in range(3)][-1]
    finally:
        recognizer.close()
    assert [h.gesture for h in hands] == [gesture]
    x, y, w, h = hands[0].box
    assert 0 <= x < x + w <= 1 and 0 <= y < y + h <= 1


# ---------------------------------------------------------- motion_detect


def test_gestures_flag_is_off_by_default():
    assert m.parse_args([]).gestures is False
    assert m.parse_args(["--gestures"]).gestures is True


def test_run_ignores_motion_while_disarmed(monkeypatch, capsys, fast_loop):  # noqa: F811
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    engine = g.GestureEngine(ScriptedRecognizer([]), rate=0.0)
    engine.armed = False
    assert m.run(0, govee_enabled=False, max_seconds=None, headless=True, gestures=engine) == 1
    assert "MOTION" not in capsys.readouterr().out
    assert engine.closed


def test_run_prints_gesture_events(monkeypatch, capsys, fast_loop):  # noqa: F811
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    engine = g.GestureEngine(ScriptedRecognizer([[hand("Open_Palm")]] * 500), rate=0.0, hold=0.0)
    m.run(0, govee_enabled=False, max_seconds=None, headless=True, gestures=engine)
    assert "GESTURE" in capsys.readouterr().out and engine.armed is False


# ------------------------------------------------------------- brightness dial


def test_brightness_from_height_maps_high_hand_to_bright():
    assert g.brightness_from_height(0.0) == 100.0
    assert g.brightness_from_height(g.BRIGHTNESS_TOP) == 100.0
    assert g.brightness_from_height(0.5) == pytest.approx(50.0)
    assert g.brightness_from_height(1.0) == 1.0


def test_trigger_active_only_after_the_hold_fired():
    trigger = g.HoldTrigger(hold=1.0, grace=0.3)
    trigger.update("Pointing_Up", 0.0)
    assert trigger.active is None
    trigger.update("Pointing_Up", 1.0)
    assert trigger.active == "Pointing_Up"
    trigger.update(None, 2.0)
    assert trigger.active is None


def test_brightness_dial_follows_the_hand_while_held():
    clock = FakeClock()
    high = hand("Pointing_Up", box=(0.4, 0.1, 0.2, 0.1))  # centre y 0.15 -> 100 %
    low = hand("Pointing_Up", box=(0.4, 0.8, 0.2, 0.1))  # centre y 0.85 -> 1 %
    engine = engine_with([[high]] * 13 + [[low]] * 20, clock, hold=1.0)
    events = hold(engine, clock, 3.0)
    assert {e.action for e in events} == {"brightness"}
    assert events[0].value == 100 and events[0].repeat is False
    assert all(e.repeat for e in events[1:])
    assert events[-1].value <= 2  # smoothed, but it gets there
    values = [e.value for e in events]
    assert values == sorted(values, reverse=True)
    assert engine.banner().startswith("Brightness")


def test_brightness_dial_ends_when_the_hand_drops():
    clock = FakeClock()
    engine = engine_with([[hand("Pointing_Up")]] * 13, clock, hold=1.0)
    hold(engine, clock, 1.2)
    assert engine.level is not None
    hold(engine, clock, 1.0)  # no hands any more
    assert engine.level is None


def test_default_bindings_only_name_shows_that_exist():
    import light_shows

    for action in g.GESTURE_BINDINGS.values():
        if action.startswith("show:"):
            assert action.removeprefix("show:") in light_shows.SHOWS


class FakeLight:
    def __init__(self) -> None:
        self.calls = []

    def play(self, show):
        self.calls.append(("play", show))

    def set_brightness(self, percent):
        self.calls.append(("brightness", percent))

    def off(self):
        self.calls.append(("off",))


@pytest.mark.parametrize(
    "event, call",
    [
        (g.Event("Victory", "show:rainbow"), ("play", "rainbow")),
        (g.Event("Closed_Fist", "arm"), ("play", "armed")),
        (g.Event("Open_Palm", "disarm"), ("play", "disarmed")),
        (g.Event("Thumb_Down", "off"), ("off",)),
        (g.Event("Wave", "snooze"), ("play", "snooze")),
        (g.Event("Snooze", "wake"), ("play", "armed")),
        (g.Event("Pointing_Up", "brightness", 42), ("brightness", 42)),
    ],
)
def test_apply_gesture_drives_the_light(event, call):
    light = FakeLight()
    m.apply_gesture(event, light)
    assert light.calls == [call]


def test_apply_gesture_ignores_unknown_actions():
    light = FakeLight()
    m.apply_gesture(g.Event("Victory", "party"), light)
    assert light.calls == []


# --------------------------------------------------------------------- wave


def feed(detector: g.WaveDetector, xs, start: float = 0.0, step: float = 0.1) -> list[bool]:
    return [detector.update(x, start + i * step) for i, x in enumerate(xs)]


WAVE_XS = [0.3, 0.45, 0.6, 0.45, 0.3, 0.45, 0.6, 0.45, 0.3]  # right, left, right, left


def test_wave_fires_once_after_enough_strokes():
    detector = g.WaveDetector(swings=4, travel=0.08, window=2.5)
    assert feed(detector, WAVE_XS) == [False] * 7 + [True, False]


def test_wave_ignores_a_still_or_drifting_hand():
    detector = g.WaveDetector(swings=4, travel=0.08, window=2.5)
    assert not any(feed(detector, [0.5, 0.51, 0.49, 0.5, 0.52, 0.5] * 5))  # jitter
    detector.reset()
    assert not any(feed(detector, [0.1 + 0.03 * i for i in range(25)]))  # one long sweep
    assert not detector.swinging


def test_wave_must_fit_in_the_time_window():
    detector = g.WaveDetector(swings=4, travel=0.08, window=2.5)
    assert not any(feed(detector, WAVE_XS, step=1.0))  # same path, far too slow


def test_wave_survives_a_short_dropout_but_not_a_long_one():
    detector = g.WaveDetector(swings=4, travel=0.08, window=2.5, grace=0.5)
    blurred = [0.3, 0.45, 0.6, None, None, 0.3, 0.45, 0.6, 0.45, 0.3]
    assert any(feed(detector, blurred))
    detector.reset()
    assert not any(feed(detector, [0.3, 0.6, 0.3] + [None] * 8 + [0.6, 0.3]))


def test_wave_reports_swinging_after_there_and_back():
    detector = g.WaveDetector(swings=4, travel=0.08, window=2.5)
    feed(detector, [0.3, 0.6])
    assert not detector.swinging  # one stroke: could be a hand arriving
    detector.update(0.3, 0.3)
    assert detector.swinging


def wave_script(gesture="Open_Palm", xs=WAVE_XS) -> list[list[g.Hand]]:
    return [[hand(gesture, box=(x - 0.1, 0.4, 0.2, 0.2))] for x in xs]


def test_engine_wave_snoozes_then_arms_again_by_itself():
    clock = FakeClock()
    engine = engine_with(wave_script(), clock)
    events = hold(engine, clock, 1.0)
    assert [(e.gesture, e.action) for e in events] == [("Wave", "snooze")]
    assert engine.armed and not engine.watching  # still armed, just not watching
    assert 0 < engine.snooze_left() <= g.SNOOZE_SECONDS
    assert "SNOOZE" in engine.banner()
    clock.now += g.SNOOZE_SECONDS
    assert [(e.gesture, e.action) for e in engine.process(FRAME)] == [("Snooze", "wake")]
    assert engine.watching and engine.snooze_left() == 0


def test_engine_waving_palm_does_not_disarm():
    clock = FakeClock()
    still_after = [[hand("Open_Palm")] for _ in range(10)]  # the hand lingers 1 s after the wave
    engine = engine_with(wave_script() + still_after, clock)
    events = hold(engine, clock, 3.0)
    assert [e.action for e in events] == ["snooze"]
    assert engine.armed


def test_engine_fist_ends_a_snooze_early():
    clock = FakeClock()
    engine = engine_with(wave_script() + [[]] * 20 + [[hand("Closed_Fist")]] * 12, clock)
    events = hold(engine, clock, 5.0)
    assert [e.action for e in events] == ["snooze", "arm"]
    assert engine.watching and engine.snooze_left() == 0


def test_engine_wave_does_nothing_while_disarmed():
    clock = FakeClock()
    engine = engine_with(wave_script(), clock)
    engine.armed = False
    assert hold(engine, clock, 1.5) == []
    assert engine.snooze_until is None


def test_engine_sideways_drift_while_dialling_is_not_a_wave():
    clock = FakeClock()
    pointing = [[hand("Pointing_Up")]] * 12
    engine = engine_with(pointing + wave_script("Pointing_Up"), clock)
    events = hold(engine, clock, 2.5)
    assert {e.action for e in events} == {"brightness"}


def test_run_ignores_motion_while_snoozed(monkeypatch, capsys, fast_loop):  # noqa: F811
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    engine = g.GestureEngine(ScriptedRecognizer([]), rate=0.0)
    engine.snooze_until = engine.clock() + 600
    assert m.run(0, govee_enabled=False, max_seconds=None, headless=True, gestures=engine) == 1
    assert "MOTION" not in capsys.readouterr().out


def test_draw_gestures_shows_the_snooze_countdown():
    clock = FakeClock()
    engine = engine_with([], clock)
    engine.snooze_until = clock.now + 125
    plain, snoozed = FRAME.copy(), np.zeros((360, 640, 3), np.uint8)
    g.draw_gestures(snoozed, engine)
    assert snoozed.any() and not plain.any()


# ------------------------------------------------------- feed (--serve) state


def test_feed_has_no_gesture_state_without_hand_signals():
    feed = m.MotionFeed()
    feed.update([], note=None, size=(1280, 720), light=None)
    state = feed.snapshot()
    assert state["mode"] is None and state["snooze"] is None
    assert state["banner"] is None and state["hands"] == []


def test_feed_carries_mode_banner_and_hands():
    clock = FakeClock()
    engine = engine_with([[hand("Open_Palm")]] * 30, clock, hold=1.0)
    feed = m.MotionFeed(clock=clock)
    engine.process(FRAME)
    feed.update([], None, (1280, 720), None, engine)
    state = feed.snapshot()
    assert state["mode"] == "armed" and state["snooze"] is None
    assert state["hands"] == [[0.4, 0.4, 0.2, 0.2, "Open Palm"]]
    hold(engine, clock, 1.5)
    feed.update([], None, (1280, 720), None, engine)
    state = feed.snapshot()
    assert state["mode"] == "disarmed"
    assert state["banner"] == "DISARMED by Open Palm"


def test_feed_counts_the_snooze_down_without_waking_viewers_every_frame():
    clock = FakeClock()
    engine = engine_with([], clock)
    engine.snooze_until = clock.now + 125
    feed = m.MotionFeed(clock=clock)
    feed.update([], None, (1280, 720), None, engine)
    version = feed.version
    assert feed.snapshot()["mode"] == "snoozed" and feed.snapshot()["snooze"] == 125
    clock.now += 5
    feed.update([], None, (1280, 720), None, engine)
    assert feed.version == version  # the countdown alone is not a change
    assert feed.snapshot()["snooze"] == 120
    engine.snooze_until = None
    feed.update([], None, (1280, 720), None, engine)
    assert feed.snapshot()["mode"] == "armed" and feed.snapshot()["snooze"] is None


def test_run_publishes_no_motion_boxes_while_disarmed(monkeypatch, capsys, fast_loop):  # noqa: F811
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    engine = g.GestureEngine(ScriptedRecognizer([]))
    engine.armed = False
    feed = m.MotionFeed()
    assert m.run(0, govee_enabled=False, max_seconds=None, headless=True, feed=feed, gestures=engine) == 1
    state = feed.snapshot()
    assert state["mode"] == "disarmed"
    assert state["since"] is None and state["boxes"] == []
