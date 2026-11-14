"""Offline tests for motion_detect. Run:
uv run --with-requirements requirements.txt --with pytest pytest -q
"""

import numpy as np
import pytest

import motion_detect as m

H, W = 720, 1280  # full-frame size; detector downscales to PROCESS_WIDTH


def background(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = np.full((H, W, 3), 90, np.int16)
    noise = rng.integers(-3, 4, size=frame.shape)  # mild sensor noise
    return np.clip(frame + noise, 0, 255).astype(np.uint8)


def trained_detector(min_area: float = m.MIN_CONTOUR_AREA) -> m.MotionDetector:
    detector = m.MotionDetector(min_area=min_area)
    for i in range(60):
        detector.detect(background(i))
    return detector


def test_static_scene_has_no_motion():
    detector = trained_detector()
    assert detector.detect(background(999)) == []


def test_large_moving_object_is_boxed_in_full_frame_coords():
    detector = trained_detector()
    frame = background(1000)
    frame[200:400, 500:700] = 255  # 200x200 bright square
    boxes = detector.detect(frame)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    # Box covers the square (dilation grows it a little), reported at 1280x720 scale.
    assert x <= 500 and y <= 200 and x + w >= 700 and y + h >= 400
    assert w < 300 and h < 300


def test_tiny_change_is_ignored_by_min_contour_area():
    detector = trained_detector()
    frame = background(1001)
    frame[300:310, 600:610] = 255  # 10x10 px speck
    assert detector.detect(frame) == []


def test_min_area_threshold_is_configurable():
    frame = background(1002)
    frame[300:340, 600:640] = 255  # 40x40 full-frame = 20x20 at 640 width
    assert trained_detector(min_area=5000).detect(frame.copy()) == []
    assert len(trained_detector(min_area=100).detect(frame.copy())) == 1


def test_cooldown_blocks_repeats_inside_window():
    now = [100.0]
    cooldown = m.Cooldown(3.0, clock=lambda: now[0])
    assert cooldown.ready() is True
    now[0] = 101.0
    assert cooldown.ready() is False
    now[0] = 102.99
    assert cooldown.ready() is False
    now[0] = 103.0
    assert cooldown.ready() is True
    now[0] = 104.0
    assert cooldown.ready() is False


@pytest.mark.parametrize(
    "light_on, last_motion, now, expected",
    [
        (None, None, 50.0, None),  # no motion yet: never touch the strip
        (None, 49.0, 50.0, "on"),  # first motion
        (False, 49.0, 50.0, "on"),
        (True, 49.0, 50.0, None),  # already on, motion continues
        (True, 30.0, 50.0, "off"),  # hold (10 s) elapsed
        (True, 40.0, 50.0, "off"),  # exactly at hold boundary
        (False, 30.0, 50.0, None),  # already off
        (None, 30.0, 50.0, None),  # unknown state, stale motion: leave it alone
    ],
)
def test_light_action(light_on, last_motion, now, expected):
    assert m.light_action(light_on, last_motion, now, hold=10.0) == expected


def test_config_defaults():
    assert m.COOLDOWN_SECONDS == 3.0
    assert m.GOVEE_DEVICE_NAME == "H6147"
    assert m.GOVEE_COLOR_VARIANT in m.VARIANTS_BY_NAME
    assert m.parse_args([]).camera == m.CAMERA_INDEX
    assert m.parse_args([]).hold == m.LIGHT_HOLD_SECONDS
    assert m.parse_args(["--hold", "4"]).hold == 4.0


def test_whole_scene_lighting_change_is_not_motion():
    # The Govee strip switching on tints the whole room: that is not motion.
    detector = trained_detector()
    frame = background(1003)
    frame[:, :, 2] = 200  # entire frame turns red-ish (BGR)
    assert detector.detect(frame) == []


def test_fast_learning_absorbs_new_lighting():
    detector = trained_detector()
    lit = background(1004)
    lit[:, :, 2] = 200
    for i in range(20):  # settle window after a light command
        detector.detect(lit, learning_rate=m.SETTLE_LEARNING_RATE)
    still_lit = background(1005)
    still_lit[:, :, 2] = 200
    assert detector.detect(still_lit) == []
    still_lit[200:400, 500:700] = 255  # real motion in the new lighting is still caught
    assert len(detector.detect(still_lit)) == 1


@pytest.mark.parametrize(
    "last_change, now, expected",
    [(None, 50.0, False), (49.0, 50.0, True), (48.0, 50.0, False)],
)
def test_light_settling_window(last_change, now, expected):
    assert m.light_settling(last_change, now, settle=2.0) is expected
