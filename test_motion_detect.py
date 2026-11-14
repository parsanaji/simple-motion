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


# ------------------------------------------------------------ camera sources


@pytest.mark.parametrize(
    "value, expected",
    [
        ("0", 0),
        ("2", 2),
        ("rtsp://127.0.0.1:8554/cam", "rtsp://127.0.0.1:8554/cam"),
        ("http://cam.local/video.mjpg", "http://cam.local/video.mjpg"),
    ],
)
def test_camera_source_accepts_index_or_url(value, expected):
    assert m.parse_camera(value) == expected
    assert m.parse_args(["--camera", value]).camera == expected


def test_camera_source_rejects_garbage():
    with pytest.raises(SystemExit):
        m.parse_args(["--camera", "webcam"])


def test_is_stream():
    assert m.is_stream("rtsp://127.0.0.1:8554/cam")
    assert not m.is_stream(0)


def test_headless_flag():
    assert m.parse_args([]).headless is False
    assert m.parse_args(["--headless"]).headless is True


# ------------------------------------------------------------ run loop


class FakeCapture:
    """Delivers the given frames, then fails every read (like an unplugged cam)."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def get(self, prop):
        return {m.cv2.CAP_PROP_FRAME_WIDTH: W, m.cv2.CAP_PROP_FRAME_HEIGHT: H}.get(prop, 0)

    def release(self):
        self.released = True


def moving_clip():
    frames = [background(i) for i in range(60)]
    for i in range(5):
        frame = background(2000 + i)
        frame[200:400, 500 + 20 * i : 700 + 20 * i] = 255
        frames.append(frame)
    return frames


def no_gui(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("headless run touched the GUI")

    for name in ("namedWindow", "imshow", "waitKey", "destroyAllWindows"):
        monkeypatch.setattr(m.cv2, name, boom)


@pytest.fixture
def fast_loop(monkeypatch):
    monkeypatch.setattr(m, "WARMUP_SECONDS", 0.0)
    monkeypatch.setattr(m, "READ_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(m, "STREAM_RECONNECT_SECONDS", 0.0)


def test_headless_run_detects_motion_without_any_window(monkeypatch, capsys, fast_loop):
    no_gui(monkeypatch)
    cap = FakeCapture(moving_clip())
    monkeypatch.setattr(m, "open_camera", lambda source: cap)
    code = m.run(0, govee_enabled=False, max_seconds=None, headless=True)
    out = capsys.readouterr().out
    assert code == 1  # local camera stopped delivering frames: exit
    assert "MOTION" in out
    assert cap.released


def test_stream_reconnects_instead_of_exiting(monkeypatch, capsys, fast_loop):
    no_gui(monkeypatch)

    class Done(Exception):
        pass

    opened = []

    def fake_open(source):
        opened.append(source)
        if len(opened) == 3:
            raise Done
        return FakeCapture(moving_clip() if len(opened) == 1 else [background(7)] * 3)

    monkeypatch.setattr(m, "open_camera", fake_open)
    url = "rtsp://127.0.0.1:8554/cam"
    with pytest.raises(Done):
        m.run(url, govee_enabled=False, max_seconds=None, headless=True)
    assert opened == [url, url, url]
    out = capsys.readouterr().out
    assert "MOTION" in out
    assert "reconnecting" in out


def test_describe_masks_stream_password():
    assert m.describe(0) == "camera 0"
    assert m.describe("rtsp://127.0.0.1:8554/cam") == "stream rtsp://127.0.0.1:8554/cam"
    masked = m.describe("rtsp://cam:s3cret@10.0.0.5:8554/cam")
    assert "s3cret" not in masked
    assert masked == "stream rtsp://cam:***@10.0.0.5:8554/cam"


# ------------------------------------------------------------ motion feed (--serve)

import json  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402


@pytest.mark.parametrize(
    "value, expected",
    [("8890", ("0.0.0.0", 8890)), ("127.0.0.1:9000", ("127.0.0.1", 9000)), (" 0 ", ("0.0.0.0", 0))],
)
def test_parse_serve(value, expected):
    assert m.parse_serve(value) == expected


@pytest.mark.parametrize("value", ["", "abc", "70000", "host:", "-1"])
def test_parse_serve_rejects_garbage(value):
    with pytest.raises(m.argparse.ArgumentTypeError):
        m.parse_serve(value)


def test_serve_flag_defaults_off():
    assert m.parse_args([]).serve is None
    assert m.parse_args(["--serve", "8890"]).serve == ("0.0.0.0", 8890)


def test_feed_normalises_boxes_to_frame_size():
    now = [50.0]
    feed = m.MotionFeed(clock=lambda: now[0])
    feed.update([(640, 360, 320, 180)], note=None, size=(1280, 720), light=None)
    state = feed.snapshot()
    assert state["motion"] is True
    assert state["boxes"] == [[0.5, 0.5, 0.25, 0.25]]
    assert state["size"] == [1280, 720]
    assert state["since"] == 0.0
    now[0] = 62.5
    feed.update([], note=None, size=(1280, 720), light=True)
    state = feed.snapshot()
    assert state["motion"] is False
    assert state["boxes"] == []
    assert state["since"] == 12.5  # seconds since the last motion
    assert state["light"] is True


def test_feed_before_any_motion():
    feed = m.MotionFeed()
    state = feed.snapshot()
    assert state["motion"] is False
    assert state["since"] is None
    assert state["note"] is None


def test_feed_version_bumps_only_on_change():
    feed = m.MotionFeed()
    feed.update([], note="warming up", size=(1280, 720), light=None)
    v1 = feed.version
    feed.update([], note="warming up", size=(1280, 720), light=None)
    assert feed.version == v1  # identical state: SSE clients are not woken
    feed.update([], note=None, size=(1280, 720), light=None)
    assert feed.version == v1 + 1


@pytest.fixture
def served_feed():
    feed = m.MotionFeed()
    server = m.start_feed_server(feed, ("127.0.0.1", 0))
    host, port = server.server_address[:2]
    yield feed, f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def test_server_motion_json_has_cors(served_feed):
    feed, base = served_feed
    feed.update([(0, 0, 128, 72)], note=None, size=(1280, 720), light=None)
    with urllib.request.urlopen(base + "/motion", timeout=5) as res:
        assert res.headers["Access-Control-Allow-Origin"] == "*"
        assert res.headers["Content-Type"].startswith("application/json")
        state = json.load(res)
    assert state["motion"] is True
    assert state["boxes"] == [[0.0, 0.0, 0.1, 0.1]]


def test_server_streams_events(served_feed):
    feed, base = served_feed
    with urllib.request.urlopen(base + "/events", timeout=5) as res:
        assert res.headers["Content-Type"].startswith("text/event-stream")
        assert res.headers["Access-Control-Allow-Origin"] == "*"
        first = json.loads(res.readline().decode().removeprefix("data: "))
        assert res.readline() == b"\n"
        assert first["motion"] is False
        feed.update([(0, 0, 640, 720)], note=None, size=(1280, 720), light=None)
        second = json.loads(res.readline().decode().removeprefix("data: "))
    assert second["motion"] is True
    assert second["boxes"] == [[0.0, 0.0, 0.5, 1.0]]


def test_server_unknown_path_is_404(served_feed):
    _, base = served_feed
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(base + "/nope", timeout=5)
    assert err.value.code == 404


def test_run_publishes_motion_to_feed(monkeypatch, capsys, fast_loop):
    no_gui(monkeypatch)
    monkeypatch.setattr(m, "open_camera", lambda source: FakeCapture(moving_clip()))
    feed = m.MotionFeed()
    assert m.run(0, govee_enabled=False, max_seconds=None, headless=True, feed=feed) == 1
    state = feed.snapshot()
    assert state["since"] is not None  # motion was seen and published
    assert state["size"] == [W, H]
