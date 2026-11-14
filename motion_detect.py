#!/usr/bin/env python3
"""Webcam motion detector (OpenCV MOG2) that lights a Govee strip on motion.

Shows the live feed, draws boxes around moving areas, prints
"MOTION <timestamp>" at most once per COOLDOWN_SECONDS, and turns the Govee
strip on (solid colour) while motion continues. The strip turns off again after
LIGHT_HOLD_SECONDS without motion.

Setup (once, in this folder):
    macOS:    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    Windows:  py -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt
Run (macOS shown; on Windows use .venv\\Scripts\\python):
    .venv/bin/python motion_detect.py
    .venv/bin/python motion_detect.py --list-cameras   # find the Logitech's index
    .venv/bin/python motion_detect.py --no-govee       # camera only, no Bluetooth
    .venv/bin/python motion_detect.py --camera 1       # override CAMERA_INDEX
    .venv/bin/python motion_detect.py --hold 30        # override LIGHT_HOLD_SECONDS
    .venv/bin/python motion_detect.py --camera rtsp://127.0.0.1:8554/cam --headless
                                                       # network stream, no window
    .venv/bin/python motion_detect.py --serve 8890     # boxes + motion state as
                                                       # JSON / server-sent events
Keys:   q (or Esc) in the video window quits; Ctrl+C quits a --headless run.
"""

# =============================================================================
# CONFIG
# =============================================================================
CAMERA_INDEX = 0  # 0 = first camera. On a Mac the built-in camera is usually 0
#                   and a USB Logitech usually 1; on a Windows PC with no
#                   built-in camera the Logitech is usually 0.
#                   Run with --list-cameras to check. --camera also takes a
#                   stream URL (rtsp://, http://), e.g. a camera that is
#                   already shared by an RTSP server.
FRAME_WIDTH = 1280  # requested capture size; the camera may pick the nearest
FRAME_HEIGHT = 720

MIN_CONTOUR_AREA = 800  # px^2, measured on the frame downscaled to
#                         PROCESS_WIDTH (640). Smaller moving blobs are ignored.
COOLDOWN_SECONDS = 3.0  # minimum gap between "MOTION" prints
WARMUP_SECONDS = 2.0  # ignore motion while MOG2 learns the scene and the
#                       camera's auto-exposure settles
MAX_CHANGED_FRACTION = 0.5  # if more than this share of the frame changes at
#                             once it is a lighting change, not motion

GOVEE_ENABLED = True
GOVEE_DEVICE_NAME = "H6147"  # matched case-insensitively inside the BLE name
#                              (e.g. "ihoment_H6147_1A2B"); never by address
GOVEE_COLOR = (255, 0, 0)  # RGB shown on motion
GOVEE_COLOR_VARIANT = "rgb-02"  # packet format confirmed working on the H6147
LIGHT_HOLD_SECONDS = 10.0  # strip turns off after this long without motion
LIGHT_SETTLE_SECONDS = 2.0  # ignore motion right after the strip switches, so
#                             its own glow on the room never counts as motion
# =============================================================================

import argparse  # noqa: E402  (imports sit below the config block on purpose)
import asyncio  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from contextlib import suppress  # noqa: E402
from datetime import datetime  # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402
from typing import Callable  # noqa: E402
from urllib.parse import urlsplit, urlunsplit  # noqa: E402

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # quiet camera-probe noise
# RTSP over TCP: no lost UDP packets (grey smears that look like motion), and
# many RTSP servers only allow TCP anyway.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2  # noqa: E402
from bleak import BleakClient, BleakScanner  # noqa: E402
from bleak.exc import BleakError  # noqa: E402

from govee_h6147 import (  # noqa: E402
    NOTIFY_CHAR,
    VARIANTS_BY_NAME,
    WRITE_CHAR,
    keepalive_packet,
    model_filter,
    power_packet,
)

PROCESS_WIDTH = 640  # detection runs on a downscaled copy: faster, and the
#                      area threshold does not depend on camera resolution
SETTLE_LEARNING_RATE = 0.2  # MOG2 re-learns the newly lit scene in ~1 s
WINDOW_NAME = "Motion (q to quit)"
MAX_READ_FAILURES = 30
READ_RETRY_SECONDS = 0.03  # pause between failed reads
STREAM_RECONNECT_SECONDS = 5.0  # a dropped stream is reopened after this pause

FEED_MAX_RATE = 15.0  # --serve: at most this many events per second per client
FEED_HEARTBEAT_SECONDS = 1.0  # resend the state this often even if unchanged

BLE_SCAN_TIMEOUT = 10.0
BLE_RECONNECT_DELAY = 5.0
BLE_KEEPALIVE_SECONDS = 2.0
BLE_TICK_SECONDS = 0.5

Box = tuple[int, int, int, int]  # x, y, w, h in full-frame pixels
Source = int | str  # camera index or stream URL


class CameraError(Exception):
    """The camera or stream could not be opened."""


# --------------------------------------------------------------------- camera


def parse_camera(value: str) -> Source:
    """--camera value: a camera index ("0") or a stream URL ("rtsp://...")."""
    if value.strip().isdigit():
        return int(value)
    if "://" in value:
        return value
    raise argparse.ArgumentTypeError(f"expected a camera index or a stream URL, got {value!r}")


def is_stream(source: Source) -> bool:
    return isinstance(source, str)


def describe(source: Source) -> str:
    """Printable name for a source, with any password in the URL masked."""
    if not is_stream(source):
        return f"camera {source}"
    parts = urlsplit(source)
    if parts.password:
        host = parts.netloc.rsplit("@", 1)[1]
        source = urlunsplit(parts._replace(netloc=f"{parts.username}:***@{host}"))
    return f"stream {source}"


def camera_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        # DirectShow opens Logitech cams in ~1 s; the MSMF default can take 30 s+.
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


MAC_PERMISSION_WAIT = 20.0  # seconds to wait for the user to click "Allow"


def _try_open(index: int) -> cv2.VideoCapture:
    backend = camera_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(index)  # let OpenCV pick any backend
    return cap


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open {describe(url)}. Check that the stream server is "
            "running and that the URL plays in VLC or ffplay."
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low where supported
    return cap


def open_camera(source: Source, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> cv2.VideoCapture:
    if is_stream(source):
        return open_stream(source)
    index = source
    cap = _try_open(index)
    if not cap.isOpened() and sys.platform == "darwin":
        # First run: macOS shows the camera permission prompt asynchronously
        # and OpenCV fails immediately. Give the user time to click Allow.
        print(f"Waiting up to {MAC_PERMISSION_WAIT:.0f}s for camera permission...", flush=True)
        deadline = time.monotonic() + MAC_PERMISSION_WAIT
        while not cap.isOpened() and time.monotonic() < deadline:
            cap.release()
            time.sleep(1.0)
            cap = _try_open(index)
    if not cap.isOpened():
        raise CameraError(
            f"Could not open camera {index}. Try --list-cameras. On macOS, allow "
            "camera access for your terminal app (System Settings > Privacy & "
            "Security > Camera); on Windows, close other apps using the camera."
        )
    if sys.platform.startswith("win"):
        # MJPG lets USB webcams deliver 720p/1080p at full frame rate.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def list_cameras(max_index: int = 6) -> None:
    print("Probing camera indexes (OpenCV cannot read camera names)...")
    found = False
    for index in range(max_index):
        cap = cv2.VideoCapture(index, camera_backend())
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok and frame is not None:
            found = True
            h, w = frame.shape[:2]
            print(f"  camera {index}: {w}x{h}")
    if not found:
        print("  none found (check camera permission / cable)")


# --------------------------------------------------------------------- motion


class MotionDetector:
    """MOG2 background subtraction; returns bounding boxes of moving areas."""

    def __init__(
        self,
        min_area: float = MIN_CONTOUR_AREA,
        process_width: int = PROCESS_WIDTH,
        max_changed_fraction: float = MAX_CHANGED_FRACTION,
    ) -> None:
        self.min_area = min_area
        self.process_width = process_width
        self.max_changed_fraction = max_changed_fraction
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=True
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame, learning_rate: float = -1.0) -> list[Box]:
        """learning_rate -1 = MOG2 automatic; higher re-learns the background faster."""
        width = frame.shape[1]
        scale = min(1.0, self.process_width / width)
        small = (
            cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else frame
        )
        small = cv2.GaussianBlur(small, (5, 5), 0)
        mask = self.subtractor.apply(small, learningRate=learning_rate)
        # MOG2 marks shadows as 127 and foreground as 255: keep only real motion.
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask) > self.max_changed_fraction * mask.size:
            return []  # whole-scene change: lights switched / exposure jump
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # drop speckle
        mask = cv2.dilate(mask, self.kernel, iterations=2)  # merge nearby blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[Box] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((round(x / scale), round(y / scale), round(w / scale), round(h / scale)))
        return boxes


class Cooldown:
    """ready() is True at most once per `seconds`."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self.clock = clock
        self._last: float | None = None

    def ready(self) -> bool:
        now = self.clock()
        if self._last is None or now - self._last >= self.seconds:
            self._last = now
            return True
        return False


def draw_overlay(frame, boxes: list[Box], note: str | None, govee_status: str | None) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if note:
        cv2.putText(frame, note, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)
    elif boxes:
        cv2.putText(frame, "MOTION", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    footer = "q: quit" + (f"   govee: {govee_status}" if govee_status else "")
    cv2.putText(
        frame, footer, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )


# --------------------------------------------------------------------- feed


def parse_serve(value: str) -> tuple[str, int]:
    """--serve value: "PORT" (all interfaces) or "HOST:PORT"."""
    host, _, port = value.strip().rpartition(":")
    if not port.isdigit() or int(port) > 65535:
        raise argparse.ArgumentTypeError(f"expected PORT or HOST:PORT, got {value!r}")
    return host or "0.0.0.0", int(port)


class MotionFeed:
    """Latest detection result, shared with the --serve HTTP threads.

    Boxes are published as fractions of the frame (x, y, w, h in 0..1), so a
    viewer can draw them over the video at whatever size it shows it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.version = 0  # bumped on every change; SSE clients wait on it
        self.closed = False
        self._cond = threading.Condition()
        self._state: dict = {"boxes": [], "note": None, "size": None, "light": None}
        self._last_motion: float | None = None

    def update(self, boxes: list[Box], note: str | None, size: tuple[int, int], light: bool | None) -> None:
        width, height = size
        state = {
            "boxes": [
                [round(x / width, 4), round(y / height, 4), round(w / width, 4), round(h / height, 4)]
                for x, y, w, h in boxes
            ],
            "note": note,
            "size": [width, height],
            "light": light,
        }
        with self._cond:
            if boxes:
                self._last_motion = self.clock()
            if state != self._state:
                self._state = state
                self.version += 1
                self._cond.notify_all()

    def snapshot(self) -> dict:
        with self._cond:
            since = None if self._last_motion is None else round(self.clock() - self._last_motion, 1)
            return {"motion": bool(self._state["boxes"]), **self._state, "since": since}

    def wait(self, seen: int, timeout: float) -> None:
        """Block until the state changes from version `seen` (or timeout / close)."""
        with self._cond:
            self._cond.wait_for(lambda: self.version != seen or self.closed, timeout)

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()


def start_feed_server(feed: MotionFeed, address: tuple[str, int]) -> ThreadingHTTPServer:
    """Serve the feed on a background thread.

    GET /motion  one JSON snapshot
    GET /events  server-sent events: one JSON snapshot per change (rate
                 limited) plus a heartbeat every FEED_HEARTBEAT_SECONDS
    CORS is open so a page served from another port (the camera page) can read it.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args) -> None:  # keep the log for motion, not HTTP
            pass

        def _start(self, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/motion":
                body = json.dumps(feed.snapshot()).encode()
                self._start("application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/events":
                self._start("text/event-stream")
                self.end_headers()
                self._stream()
            else:
                self.send_error(404)

        def _stream(self) -> None:
            with suppress(OSError):  # viewer went away (broken pipe / reset)
                while not feed.closed:
                    seen = feed.version
                    self.wfile.write(b"data: " + json.dumps(feed.snapshot()).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1.0 / FEED_MAX_RATE)
                    feed.wait(seen, FEED_HEARTBEAT_SECONDS)

    server = ThreadingHTTPServer(address, Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, name="motion-feed", daemon=True).start()
    return server


# --------------------------------------------------------------------- govee


def light_action(light_on: bool | None, last_motion: float | None, now: float, hold: float) -> str | None:
    """Decide what the strip should do. light_on None = unknown (never touched)."""
    if last_motion is None:
        return None
    want_on = now - last_motion < hold
    if want_on and light_on is not True:
        return "on"
    if not want_on and light_on is True:
        return "off"
    return None


def light_settling(last_change: float | None, now: float, settle: float = LIGHT_SETTLE_SECONDS) -> bool:
    """True while the room lighting is still adjusting to a strip on/off."""
    return last_change is not None and now - last_change < settle


class GoveeController:
    """Keeps a BLE link to the strip on a background thread with its own event loop.

    The OpenCV window must stay on the main thread (macOS requirement), so all
    Bluetooth work happens here. The main thread only calls motion() and stop().
    """

    def __init__(
        self,
        name: str = GOVEE_DEVICE_NAME,
        color: tuple[int, int, int] = GOVEE_COLOR,
        variant: str = GOVEE_COLOR_VARIANT,
        hold: float = LIGHT_HOLD_SECONDS,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.verbose = verbose
        self.color_packet = VARIANTS_BY_NAME[variant].packet(color)
        self.hold = hold
        self.status = "starting"
        self.last_change: float | None = None  # monotonic time of last on/off
        self._last_motion: float | None = None  # float writes are atomic under the GIL
        self._light_on: bool | None = None
        self._in_session = False
        self._loop = asyncio.new_event_loop()
        self._wake = asyncio.Event()
        self._stopping = asyncio.Event()
        self._main_task: asyncio.Task | None = None
        self._thread = threading.Thread(target=self._run, name="govee-ble", daemon=True)

    @property
    def light_on(self) -> bool | None:
        """True / False once the strip has been switched; None before."""
        return self._light_on

    # ---- called from the main thread
    def start(self) -> None:
        self._thread.start()

    def motion(self) -> None:
        self._last_motion = time.monotonic()
        if self._light_on is not True:  # only wake the BLE loop when it has work
            with suppress(RuntimeError):  # loop already closed during shutdown
                self._loop.call_soon_threadsafe(self._wake.set)

    def stop(self, timeout: float = 6.0) -> None:
        if not self._thread.is_alive():
            return

        def signal() -> None:
            self._stopping.set()
            self._wake.set()
            # Not connected (scanning / connecting / waiting): nothing to turn
            # off, so cancel right away instead of waiting out the scan.
            if not self._in_session and self._main_task:
                self._main_task.cancel()

        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(signal)
        self._thread.join(timeout)

    # ---- BLE thread
    def _set_status(self, status: str) -> None:
        if status != self.status:
            self.status = status
            print(f"[govee] {status}", flush=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._main_task = self._loop.create_task(self._main())
        with suppress(asyncio.CancelledError):
            self._loop.run_until_complete(self._main_task)
        self._loop.close()

    async def _main(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._session()
            except (BleakError, asyncio.TimeoutError, OSError) as exc:
                self._set_status(f"error: {exc or type(exc).__name__}")
            if not self._stopping.is_set():
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._stopping.wait(), BLE_RECONNECT_DELAY)

    async def _session(self) -> None:
        self._set_status(f"scanning for {self.name}")
        device = await BleakScanner.find_device_by_filter(
            model_filter(self.name), timeout=BLE_SCAN_TIMEOUT
        )
        if device is None:
            self._set_status(f"{self.name} not found, retrying")
            return
        self._set_status(f"connecting to {device.name}")
        async with BleakClient(device, timeout=20.0) as client:
            self._set_status(f"connected to {device.name}")
            if self.verbose:  # the strip echoes each accepted command back
                with suppress(BleakError):
                    await client.start_notify(
                        NOTIFY_CHAR,
                        lambda _s, data: print(f"[govee] <- {bytes(data).hex(' ')}", flush=True),
                    )
            self._in_session = True
            try:
                await self._control_loop(client)
            finally:
                self._in_session = False
        if not self._stopping.is_set():
            self._set_status("disconnected, reconnecting")

    async def _control_loop(self, client: BleakClient) -> None:
        async def write(packet: bytes) -> None:
            if self.verbose:
                print(f"[govee] -> {packet.hex(' ')}", flush=True)
            await client.write_gatt_char(WRITE_CHAR, packet)

        last_write = time.monotonic()
        while client.is_connected:
            self._wake.clear()
            if self._stopping.is_set():
                if self._light_on:
                    await write(power_packet(False))
                    self._light_on = False
                    print("[govee] light off (exit)", flush=True)
                    await asyncio.sleep(0.3)  # let the write flush before disconnect
                return

            now = time.monotonic()
            action = light_action(self._light_on, self._last_motion, now, self.hold)
            if action == "on":
                self.last_change = now
                await write(power_packet(True))
                await write(self.color_packet)
                self.last_change = time.monotonic()
                self._light_on = True
                print("[govee] light on", flush=True)
                last_write = now
            elif action == "off":
                self.last_change = now
                await write(power_packet(False))
                self.last_change = time.monotonic()
                self._light_on = False
                print(f"[govee] light off ({self.hold:g}s without motion)", flush=True)
                last_write = now
            elif now - last_write >= BLE_KEEPALIVE_SECONDS:
                await write(keepalive_packet())  # Govee drops idle links
                last_write = now

            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._wake.wait(), BLE_TICK_SECONDS)


# --------------------------------------------------------------------- main


def reconnect(source: Source, cap: cv2.VideoCapture) -> cv2.VideoCapture:
    """Reopen a dropped stream, retrying until it comes back."""
    cap.release()
    while True:
        print(f"{describe(source)} lost, reconnecting in {STREAM_RECONNECT_SECONDS:g}s", flush=True)
        time.sleep(STREAM_RECONNECT_SECONDS)
        try:
            cap = open_camera(source)
        except CameraError as exc:
            print(exc, file=sys.stderr, flush=True)
            continue
        print(f"{describe(source)} reconnected", flush=True)
        return cap


def run(
    camera: Source,
    govee_enabled: bool,
    max_seconds: float | None,
    hold: float = LIGHT_HOLD_SECONDS,
    verbose: bool = False,
    headless: bool = False,
    feed: MotionFeed | None = None,
) -> int:
    cap = open_camera(camera)
    detector = MotionDetector()
    cooldown = Cooldown(COOLDOWN_SECONDS)
    govee = GoveeController(hold=hold, verbose=verbose) if govee_enabled else None
    if govee:
        govee.start()

    if not headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    how = "Headless, Ctrl+C to quit." if headless else "Press q in the window to quit."
    print(f"{describe(camera).capitalize()} open at {width}x{height}. {how}", flush=True)

    started = time.monotonic()
    failures = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    if not is_stream(camera):
                        print("Camera stopped delivering frames.", file=sys.stderr)
                        return 1
                    # Keep the detector and the Bluetooth link; only the
                    # network stream needs to come back.
                    cap = reconnect(camera, cap)
                    failures = 0
                elif headless:
                    time.sleep(READ_RETRY_SECONDS)
                else:
                    cv2.waitKey(max(1, int(READ_RETRY_SECONDS * 1000)))
                continue
            failures = 0

            now = time.monotonic()
            warming = now - started < WARMUP_SECONDS
            settling = govee is not None and light_settling(govee.last_change, now)
            # Always feed MOG2 so it keeps learning; learn fast while the
            # room adjusts to the strip switching on/off.
            boxes = detector.detect(frame, SETTLE_LEARNING_RATE if settling else -1.0)
            note = "warming up..." if warming else "light settling..." if settling else None
            if note:
                boxes = []

            if feed:
                feed.update(boxes, note, (frame.shape[1], frame.shape[0]), govee.light_on if govee else None)
            if boxes:
                if govee:
                    govee.motion()
                if cooldown.ready():
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    areas = f"{len(boxes)} area" + ("s" if len(boxes) != 1 else "")
                    print(f"MOTION {stamp} ({areas})", flush=True)

            if not headless:
                draw_overlay(frame, boxes, note, govee.status if govee else None)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # q or Esc
                    return 0
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                return 0
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # macOS only closes the window after one more event pump
        if govee:
            govee.stop()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam motion detector with Govee light.")
    parser.add_argument(
        "--camera", type=parse_camera, default=CAMERA_INDEX,
        help=f"camera index or stream URL such as rtsp://host:8554/cam (default {CAMERA_INDEX})",
    )
    parser.add_argument("--list-cameras", action="store_true", help="probe camera indexes and exit")
    parser.add_argument("--no-govee", action="store_true", help="do not use the Govee strip")
    parser.add_argument("--seconds", type=float, help="quit automatically after N seconds")
    parser.add_argument(
        "--hold", type=float, default=LIGHT_HOLD_SECONDS,
        help=f"seconds without motion before the strip turns off (default {LIGHT_HOLD_SECONDS:g})",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="log every Govee packet and reply")
    parser.add_argument("--headless", action="store_true", help="no video window (for running as a background service)")
    parser.add_argument(
        "--serve", type=parse_serve, metavar="[HOST:]PORT",
        help="publish motion boxes over HTTP: /motion (JSON) and /events (server-sent events)",
    )
    return parser.parse_args(argv)


def _interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        list_cameras()
        return 0
    # A service manager stopping us (SIGTERM) should still turn the light off.
    signal.signal(signal.SIGTERM, _interrupt)
    feed = server = None
    if args.serve:
        feed = MotionFeed()
        try:
            server = start_feed_server(feed, args.serve)
        except OSError as exc:
            print(f"Cannot serve on {args.serve[0]}:{args.serve[1]}: {exc}", file=sys.stderr)
            return 2
        host, port = server.server_address[:2]
        print(f"Motion feed on http://{host}:{port}/events", flush=True)
    try:
        return run(
            args.camera, GOVEE_ENABLED and not args.no_govee, args.seconds,
            args.hold, args.verbose, args.headless, feed,
        )
    except CameraError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if server:
            feed.close()
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    sys.exit(main())
