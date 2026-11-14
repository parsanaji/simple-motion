"""Hand-signal control for motion_detect: hold a gesture to arm or disarm.

Recognition is Google's MediaPipe Gesture Recognizer (a small pretrained model,
a few milliseconds per frame on a laptop CPU). This file only adds the part
around it: "held long enough to count" and "what does that gesture do".

Setup (once, on top of the normal install):
    macOS:    .venv/bin/pip install -r requirements-gestures.txt
    Windows:  .venv\\Scripts\\pip install -r requirements-gestures.txt
Run:
    .venv/bin/python motion_detect.py --gestures
The model file (8 MB) is downloaded into models/ the first time.
"""

# =============================================================================
# CONFIG
# =============================================================================
# Gestures the model knows: Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down,
# Thumb_Up, Victory, ILoveYou. Bind any of them to an action name.
# "arm" and "disarm" are built in: while disarmed, motion is ignored.
# Other actions: "show:<name>" plays a light show from light_shows.py,
# "off" stops the show and the light, and "brightness" is a dial: keep the
# gesture up and move your hand higher (brighter) or lower (dimmer).
GESTURE_BINDINGS = {
    "Closed_Fist": "arm",
    "Open_Palm": "disarm",
    "Victory": "show:rainbow",
    "ILoveYou": "show:heartbeat",
    "Thumb_Up": "show:disco",
    "Thumb_Down": "off",
    "Pointing_Up": "brightness",
}
GESTURE_HOLD_SECONDS = 1.0  # hold a gesture this long before it counts
GESTURE_GRACE_SECONDS = 0.3  # a gesture may flicker away this long mid-hold
GESTURE_MIN_SCORE = 0.6  # 0..1, ignore guesses the model is less sure about
GESTURE_RATE = 10.0  # look for hands this many times a second (0 = every frame)
GESTURE_MAX_HANDS = 2
BRIGHTNESS_TOP = 0.3  # middle of the hand this far down the picture (0 = top edge) = 100 %
BRIGHTNESS_BOTTOM = 0.7  # this far down = 1 %. Kept well inside the picture: a hand
#                          half out of view is no longer recognised
BRIGHTNESS_SMOOTHING = 0.5  # 0..1: lower = steadier but slower to follow the hand
WAVE_ACTION = "snooze"  # what waving a hand side to side does ("" = nothing)
WAVE_SWINGS = 4  # side-to-side strokes that make a wave (right, left, right, left)
WAVE_MIN_TRAVEL = 0.08  # each stroke must cover this much of the picture width
WAVE_WINDOW_SECONDS = 2.5  # ...and all of them must fit in this long
WAVE_QUIET_SECONDS = 1.5  # after a wave, ignore held gestures this long (the hand is leaving)
SNOOZE_SECONDS = 300.0  # a snooze ignores motion this long, then arms again by itself
BANNER_SECONDS = 3.0  # how long "ARMED by ..." stays on the video
# =============================================================================

import time  # noqa: E402  (imports sit below the config block on purpose)
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable, Protocol  # noqa: E402
from urllib.request import urlopen  # noqa: E402

import cv2  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "gesture_recognizer.task"
MODEL_DOWNLOAD_TIMEOUT = 60.0

NormBox = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame
PRETTY = {"ILoveYou": "I Love You"}


class GestureError(Exception):
    """Gestures cannot start (missing package or model); the message says why."""


@dataclass(frozen=True)
class Hand:
    gesture: str | None  # None = a hand, but no known gesture
    score: float
    side: str  # "Left" / "Right" as the model sees it
    box: NormBox


@dataclass(frozen=True)
class Event:
    gesture: str
    action: str
    value: int | None = None  # "brightness": the level in percent
    repeat: bool = False  # True for the follow-up events of a dial that is being held


def brightness_from_height(y: float) -> float:
    """Hand height in the picture (0 = top, 1 = bottom) -> brightness 1..100 %."""
    level = (BRIGHTNESS_BOTTOM - y) / (BRIGHTNESS_BOTTOM - BRIGHTNESS_TOP) * 100
    return max(1.0, min(100.0, level))


def pretty(gesture: str) -> str:
    return PRETTY.get(gesture, gesture.replace("_", " "))


def best_hand(hands: list[Hand], min_score: float = GESTURE_MIN_SCORE) -> Hand | None:
    """The most confident hand that shows a known gesture."""
    shown = [h for h in hands if h.gesture and h.score >= min_score]
    return max(shown, key=lambda h: h.score, default=None)


class HoldTrigger:
    """Turns a per-frame gesture guess into one event per deliberate hold."""

    def __init__(self, hold: float = GESTURE_HOLD_SECONDS, grace: float = GESTURE_GRACE_SECONDS) -> None:
        self.hold = hold
        self.grace = grace
        self.gesture: str | None = None  # the gesture being held
        self._since = 0.0
        self._seen = 0.0
        self._fired = False

    def update(self, gesture: str | None, now: float) -> str | None:
        """Feed the current guess; returns the gesture once when the hold completes."""
        if gesture != self.gesture:
            if self.gesture and now - self._seen <= self.grace:
                return None  # brief dropout or misread: keep the hold alive
            self.gesture, self._since, self._fired = gesture, now, False
        self._seen = now
        if self.gesture and not self._fired and now - self._since >= self.hold:
            self._fired = True
            return self.gesture
        return None

    @property
    def active(self) -> str | None:
        """The gesture that has fired and is still being held."""
        return self.gesture if self._fired else None

    def progress(self, now: float) -> float:
        """0..1: how far the current hold is."""
        if not self.gesture:
            return 0.0
        if self._fired or self.hold <= 0:
            return 1.0
        return min(1.0, (now - self._since) / self.hold)


class WaveDetector:
    """Spots a hand waving side to side from its x position, look by look."""

    def __init__(
        self,
        swings: int = WAVE_SWINGS,
        travel: float = WAVE_MIN_TRAVEL,
        window: float = WAVE_WINDOW_SECONDS,
        grace: float = 0.5,
    ) -> None:
        self.swings = swings
        self.travel = travel
        self.window = window
        self.grace = grace  # a fast hand blurs: it may vanish this long mid-wave
        self.reset()

    def reset(self) -> None:
        self._edge: float | None = None  # furthest point of the current stroke
        self._direction = 0  # +1 right, -1 left, 0 not moving yet
        self._strokes: list[float] = []  # when each stroke was completed
        self._seen = 0.0

    @property
    def swinging(self) -> bool:
        """True once the hand has gone there and back: probably a wave, not a held gesture."""
        return len(self._strokes) >= 2

    def update(self, x: float | None, now: float) -> bool:
        """Feed the hand's x (0..1, None = no hand); True once when a wave completes."""
        if x is None:
            if self._edge is not None and now - self._seen > self.grace:
                self.reset()
            return False
        self._seen = now
        self._strokes = [t for t in self._strokes if now - t <= self.window]
        if self._edge is None:
            self._edge = x
            return False
        move = x - self._edge
        if self._direction and move * self._direction > 0:
            self._edge = x  # still going the same way: the stroke grows
        elif abs(move) >= self.travel:  # set off, or turned round and came back far enough
            self._direction = 1 if move > 0 else -1
            self._edge = x
            self._strokes.append(now)
            if len(self._strokes) >= self.swings:
                self.reset()
                return True
        return False


class Recognizer(Protocol):
    def recognize(self, frame, now: float) -> list[Hand]: ...

    def close(self) -> None: ...


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the model next to the script the first time it is needed."""
    if path.exists():
        return path
    print(f"Downloading the gesture model to {path} ...", flush=True)
    partial = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(MODEL_URL, timeout=MODEL_DOWNLOAD_TIMEOUT) as response:
            partial.write_bytes(response.read())
        partial.replace(path)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise GestureError(
            f"Could not download the gesture model ({exc}). Download {MODEL_URL} yourself and save it as {path}."
        ) from exc
    return path


class MediaPipeRecognizer:
    """Finds hands in a BGR frame and names the gesture each one shows."""

    def __init__(self, model: Path = MODEL_PATH, max_hands: int = GESTURE_MAX_HANDS) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise GestureError(
                "--gestures needs MediaPipe. Install it with: pip install -r requirements-gestures.txt"
            ) from exc
        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(ensure_model(model))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._stamp = -1

    def recognize(self, frame, now: float) -> list[Hand]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._stamp = max(self._stamp + 1, int(now * 1000))  # must always increase
        result = self._recognizer.recognize_for_video(image, self._stamp)
        hands = []
        for gestures, sides, points in zip(result.gestures, result.handedness, result.hand_landmarks):
            top = gestures[0]
            xs = [min(1.0, max(0.0, p.x)) for p in points]
            ys = [min(1.0, max(0.0, p.y)) for p in points]
            box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
            name = top.category_name if top.category_name not in ("", "None") else None
            hands.append(Hand(name, float(top.score), sides[0].category_name, box))
        return hands

    def close(self) -> None:
        self._recognizer.close()


class GestureEngine:
    """Watches frames for held gestures and keeps the armed / disarmed state."""

    def __init__(
        self,
        recognizer: Recognizer,
        bindings: dict[str, str] | None = None,
        hold: float = GESTURE_HOLD_SECONDS,
        grace: float = GESTURE_GRACE_SECONDS,
        min_score: float = GESTURE_MIN_SCORE,
        rate: float = GESTURE_RATE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.recognizer = recognizer
        self.bindings = dict(GESTURE_BINDINGS if bindings is None else bindings)
        self.trigger = HoldTrigger(hold, grace)
        self.min_score = min_score
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.clock = clock
        self.wave = WaveDetector()
        self.armed = True
        self.snooze_until: float | None = None  # set while a snooze is running
        self.closed = False
        self.hands: list[Hand] = []  # from the latest look, for drawing
        self.level: float | None = None  # brightness while the dial is in use
        self._due: float | None = None  # when to look next
        self._banner: tuple[str, float] | None = None
        self._quiet_until = float("-inf")

    @property
    def watching(self) -> bool:
        """Should motion count right now? Armed, and not snoozed."""
        return self.armed and self.snooze_until is None

    def snooze_left(self) -> float:
        """Seconds of snooze remaining (0 when not snoozed)."""
        return max(0.0, self.snooze_until - self.clock()) if self.snooze_until is not None else 0.0

    def process(self, frame) -> list[Event]:
        """Look at one frame; returns the actions that just fired (usually none)."""
        now = self.clock()
        # Keep a steady schedule: a camera frame that lands a hair early still
        # counts, so 30 fps video really gives GESTURE_RATE looks a second.
        if self._due is not None and now < self._due - self.interval * 0.1:
            return []
        self._due = now + self.interval if self._due is None else self._due + self.interval
        if self._due < now:  # fell behind (slow frame): restart the schedule
            self._due = now + self.interval
        self.hands = self.recognizer.recognize(frame, now)
        events: list[Event] = []
        if self.snooze_until is not None and now >= self.snooze_until:
            self.snooze_until = None
            self._banner = ("ARMED again (snooze over)", now)
            events.append(Event("Snooze", "wake"))

        dialling = self.bindings.get(self.trigger.active or "") == "brightness"
        if dialling or not WAVE_ACTION:
            self.wave.reset()  # a dial hand wanders sideways too
        elif self._waved(now):
            self._quiet_until = now + WAVE_QUIET_SECONDS
            if WAVE_ACTION != "snooze":
                self._banner = (f"{WAVE_ACTION} (Wave)", now)
                events.append(Event("Wave", WAVE_ACTION))
            elif self.armed:  # nothing to snooze while disarmed
                self.snooze_until = now + SNOOZE_SECONDS
                self._banner = (f"SNOOZE {SNOOZE_SECONDS / 60:g} min (Wave)", now)
                events.append(Event("Wave", "snooze"))
        waving = self.wave.swinging or now < self._quiet_until

        shown = None if waving else best_hand(self.hands, self.min_score)
        gesture = self.trigger.update(shown.gesture if shown else None, now)
        held = self.trigger.active
        if held and self.bindings.get(held) == "brightness":
            if not shown or shown.gesture != held:
                return events  # hand flickered away for a moment: keep the level
            target = brightness_from_height(shown.box[1] + shown.box[3] / 2)
            if self.level is None:
                self.level = target
            else:
                self.level += BRIGHTNESS_SMOOTHING * (target - self.level)
            self._banner = (f"Brightness {round(self.level)}%", now)
            return events + [Event(held, "brightness", round(self.level), repeat=gesture is None)]
        self.level = None
        action = self.bindings.get(gesture) if gesture else None
        if not action:
            return events
        if action in ("arm", "disarm"):
            self.armed = action == "arm"
            self.snooze_until = None  # either one ends a snooze
            text = f"{'ARMED' if self.armed else 'DISARMED'} by {pretty(gesture)}"
        else:
            text = f"{action} ({pretty(gesture)})"
        self._banner = (text, now)
        return events + [Event(gesture, action)]

    def _waved(self, now: float) -> bool:
        """Feed the biggest hand in view to the wave detector."""
        biggest = max(self.hands, key=lambda h: h.box[2] * h.box[3], default=None)
        return self.wave.update(biggest.box[0] + biggest.box[2] / 2 if biggest else None, now)

    def progress(self) -> float:
        return self.trigger.progress(self.clock())

    def banner(self) -> str | None:
        """Text announcing the last action, for a few seconds after it fired."""
        if self._banner and self.clock() - self._banner[1] < BANNER_SECONDS:
            return self._banner[0]
        return None

    def close(self) -> None:
        self.closed = True
        self.recognizer.close()


def create_engine() -> GestureEngine:
    """The real thing: MediaPipe plus the bindings from the CONFIG block."""
    return GestureEngine(MediaPipeRecognizer())


def draw_gestures(frame, engine: GestureEngine) -> None:
    """Draw hand boxes, the hold bar, the armed badge and the action banner."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    cyan, white = (255, 200, 0), (255, 255, 255)
    held = engine.trigger.gesture
    for hand in engine.hands:
        x, y = int(hand.box[0] * width), int(hand.box[1] * height)
        w, h = int(hand.box[2] * width), int(hand.box[3] * height)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cyan, 2)
        label = f"{pretty(hand.gesture)} {hand.score:.0%}" if hand.gesture else "hand"
        cv2.putText(frame, label, (x, max(18, y - 8)), font, 0.6, cyan, 2, cv2.LINE_AA)
        if held and hand.gesture == held and held in engine.bindings:
            filled = int(w * engine.progress())
            cv2.rectangle(frame, (x, y + h + 6), (x + w, y + h + 16), white, 1)
            cv2.rectangle(frame, (x, y + h + 6), (x + filled, y + h + 16), white, -1)

    left = engine.snooze_left()
    if left > 0:
        badge, color = f"SNOOZE {int(left) // 60}:{int(left) % 60:02d}", (0, 165, 255)
    else:
        badge = "ARMED" if engine.armed else "DISARMED"
        color = (0, 0, 255) if engine.armed else (0, 200, 0)
    (text_w, _), _ = cv2.getTextSize(badge, font, 0.9, 2)
    cv2.putText(frame, badge, (width - text_w - 12, 36), font, 0.9, color, 2, cv2.LINE_AA)

    if engine.level is not None:  # brightness dial: a gauge down the left edge
        top, bottom = int(BRIGHTNESS_TOP * height), int(BRIGHTNESS_BOTTOM * height)
        mark = bottom - int((bottom - top) * engine.level / 100)
        cv2.rectangle(frame, (16, top), (36, bottom), white, 2)
        cv2.rectangle(frame, (16, mark), (36, bottom), white, -1)

    banner = engine.banner()
    if banner:
        (text_w, _), _ = cv2.getTextSize(banner, font, 1.0, 3)
        cv2.putText(frame, banner, ((width - text_w) // 2, 84), font, 1.0, white, 3, cv2.LINE_AA)
