"""Double-clap detection for motion_detect: clap twice, the light toggles.

Setup (once, on top of the normal install):
    pip install -r requirements-claps.txt
Then:
    python motion_detect.py --claps

The microphone is only listened to for loudness; nothing is recorded or kept.
macOS asks for microphone permission for your terminal app the first time.
"""

from __future__ import annotations

# ============================== CONFIG =======================================
CLAP_THRESHOLD = 0.15  # 0..1: a clap's peak must be at least this loud (1 = microphone maximum)
CLAP_RATIO = 4.0  # ...and this many times louder than the room's background level
CLAP_MAX_LENGTH = 0.15  # seconds: a clap is short. Longer loud sounds (talk, music) are ignored
CLAP_MIN_GAP = 0.12  # seconds between the two claps, at least
CLAP_MAX_GAP = 0.6  # ...and at most
CLAP_CONFIRM_SECONDS = 0.4  # wait this long for a third clap: applause is not a double clap
CLAP_REST_SECONDS = 1.0  # after a double clap (or applause), ignore sound this long
MIC_SAMPLE_RATE = 16000
MIC_BLOCK = 256  # samples per loudness reading (16 ms)
MIC_DEVICE = None  # None = the system default microphone, or a device name / number
# =============================================================================

import threading  # noqa: E402  (imports sit below the config block on purpose)
import time  # noqa: E402


class ClapError(Exception):
    """Clap detection cannot start (missing package or microphone); the message says why."""


class ClapDetector:
    """Turns a stream of loudness readings into double-clap events. No audio code here."""

    def __init__(
        self,
        threshold: float = CLAP_THRESHOLD,
        ratio: float = CLAP_RATIO,
        max_length: float = CLAP_MAX_LENGTH,
        min_gap: float = CLAP_MIN_GAP,
        max_gap: float = CLAP_MAX_GAP,
        confirm: float = CLAP_CONFIRM_SECONDS,
        rest: float = CLAP_REST_SECONDS,
    ) -> None:
        self.threshold = threshold
        self.ratio = ratio
        self.max_length = max_length
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.confirm = confirm
        self.rest = rest
        self.background = 0.0  # slow average of the quiet level
        self._loud_since: float | None = None  # start of the loud burst under way
        self._too_long = False  # that burst is no clap
        self._first: float | None = None  # when the first clap of a pair started
        self._fire_at: float | None = None  # a pair is complete: fire then, unless a third clap comes
        self._rest_until = float("-inf")

    def feed(self, level: float, now: float) -> bool:
        """Feed one loudness reading (0..1 peak); True once per double clap."""
        loud = level >= max(self.threshold, self.ratio * self.background)
        if not loud or self._too_long:  # a noise that goes on and on is background too
            self.background += 0.05 * (level - self.background)
        if not loud:
            self._too_long = self._too_long and self._loud_since is not None
        if now < self._rest_until:
            self._loud_since = None
            return False

        if loud:
            if self._loud_since is None:
                self._loud_since, self._too_long = now, False
            elif now - self._loud_since > self.max_length and not self._too_long:
                self._too_long = True  # a long noise: forget any clap before it too
                self._first = self._fire_at = None
        elif self._loud_since is not None:
            started, self._loud_since = self._loud_since, None
            if not self._too_long:
                self._clap(started, now)

        if self._fire_at is not None and now >= self._fire_at:
            self._first = self._fire_at = None
            self._rest_until = now + self.rest
            return True
        return False

    def _clap(self, started: float, now: float) -> None:
        if self._fire_at is not None:  # a third clap: applause
            self._first = self._fire_at = None
            self._rest_until = now + self.rest
        elif self._first is not None and self.min_gap <= started - self._first <= self.max_gap:
            self._fire_at = started + self.confirm
        else:
            self._first = started


class ClapListener:
    """Listens on the microphone in the background; poll() says how many double claps came in."""

    def __init__(self, detector: ClapDetector | None = None, device=MIC_DEVICE) -> None:
        try:
            import sounddevice
        except (ImportError, OSError) as exc:  # OSError: PortAudio library missing
            raise ClapError("--claps needs sounddevice. Install it with: pip install -r requirements-claps.txt") from exc
        self.detector = detector or ClapDetector()
        self.peak = 0.0  # loudest reading since the last poll, for tuning with -v
        self._count = 0
        self._lock = threading.Lock()
        try:
            self._stream = sounddevice.InputStream(
                samplerate=MIC_SAMPLE_RATE, blocksize=MIC_BLOCK, channels=1, dtype="float32",
                device=device, callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:  # sounddevice.PortAudioError, ValueError: no usable microphone
            raise ClapError(f"Could not open the microphone ({exc}).") from exc

    def _on_audio(self, block, _frames, _time, _status) -> None:
        self.hear(float(abs(block).max()) if len(block) else 0.0, time.monotonic())

    def hear(self, level: float, now: float) -> None:
        fired = self.detector.feed(level, now)
        with self._lock:
            self.peak = max(self.peak, level)
            self._count += fired

    def poll(self) -> int:
        """Double claps since the last call."""
        with self._lock:
            count, self._count, self.peak = self._count, 0, 0.0
        return count

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
